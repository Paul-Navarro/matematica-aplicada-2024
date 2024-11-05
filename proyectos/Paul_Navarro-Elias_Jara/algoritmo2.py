import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import skfuzzy as fuzz
import time

# Descargamos el analizador de sentimiento de VADER
nltk.download('vader_lexicon')

# Función para convertir contracciones a su forma completa
def expand_contractions(text):
    contractions = {
        "can't": "can not",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    for contraction, full_form in contractions.items():
        text = re.sub(contraction, full_form, text)
    return text

# Función para limpiar el texto según la especificación
def preprocess_text(text):
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Eliminar menciones con @
    text = re.sub(r'@\w+', '', text)
    # Eliminar caracteres especiales y dígitos
    text = re.sub(r'[^A-Za-z\s#]', '', text)
    # Eliminar el símbolo # pero conservar la palabra
    text = re.sub(r'#', '', text)
    # Expandir contracciones
    text = expand_contractions(text)
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar espacios extra
    text = text.strip()
    return text

# Leer el archivo CSV con datos
df = pd.read_csv('test_data.csv')

# Aplicar el preprocesamiento a la columna 'sentence'
df['cleaned_sentence'] = df['sentence'].apply(preprocess_text)

# Inicializar el analizador de sentimientos VADER
sia = SentimentIntensityAnalyzer()

# Función para obtener los puntajes de sentimiento, incluyendo el neutral
def get_sentiment_scores(text):
    sentiment = sia.polarity_scores(text)
    return sentiment['pos'], sentiment['neg'], sentiment['neu']

# Aplicar la función de puntajes de sentimiento a cada oración del dataset
df['positive_score'], df['negative_score'], df['neutral_score'] = zip(*df['cleaned_sentence'].apply(get_sentiment_scores))

# Calcular los valores mínimo, medio y máximo para puntajes positivos y negativos
pos_min, pos_max = df['positive_score'].min(), df['positive_score'].max()
neg_min, neg_max = df['negative_score'].min(), df['negative_score'].max()
pos_mid = (pos_min + pos_max) / 2
neg_mid = (neg_min + neg_max) / 2

# Definir los parámetros para los conjuntos difusos
fuzzy_sets = {
    'positive': {
        'Low': (pos_min, pos_min, pos_mid),
        'Medium': (pos_min, pos_mid, pos_max),
        'High': (pos_mid, pos_max, pos_max)
    },
    'negative': {
        'Low': (neg_min, neg_min, neg_mid),
        'Medium': (neg_min, neg_mid, neg_max),
        'High': (neg_mid, neg_max, neg_max)
    },
    'output': {
        'Negative': (0, 0, 5),
        'Neutral': (0, 5, 10),
        'Positive': (5, 10, 10)
    }
}

# Función de pertenencia triangular
def triangular_membership(x, d, e, f):
    if x <= d:
        return 0
    elif d < x <= e:
        return (x - d) / (e - d)
    elif e < x < f:
        return (f - x) / (f - e)
    else:
        return 0

# Ajustamos la función para recibir tanto el conjunto difuso 'positive' como 'negative'
def fuzzify_score(score, positive_fuzzy_set, negative_fuzzy_set):
    # Usamos perf_counter para obtener una medición más precisa
    inicio = time.perf_counter()
    
    # Calcular la pertenencia para el puntaje positivo
    positive_low = triangular_membership(score['positive_score'], *positive_fuzzy_set['Low'])
    positive_medium = triangular_membership(score['positive_score'], *positive_fuzzy_set['Medium'])
    positive_high = triangular_membership(score['positive_score'], *positive_fuzzy_set['High'])
    
    # Calcular la pertenencia para el puntaje negativo
    negative_low = triangular_membership(score['negative_score'], *negative_fuzzy_set['Low'])
    negative_medium = triangular_membership(score['negative_score'], *negative_fuzzy_set['Medium'])
    negative_high = triangular_membership(score['negative_score'], *negative_fuzzy_set['High'])
    
    # Medimos el tiempo de finalización
    fin = time.perf_counter()
    
    # Convertimos el tiempo a milisegundos para mayor precisión
    return (fin - inicio) * 1000  # Tiempo en milisegundos

# Aplicamos la función modificada
df['tiempo_fuzzification'] = df.apply(lambda x: fuzzify_score(x, fuzzy_sets['positive'], fuzzy_sets['negative']), axis=1)

# Convertir la columna de 'sentiment' de 1/0 a "Positiva"/"Negativa"
df['sentiment'] = df['sentiment'].replace({1: "Positive", 0: "Negative"})

# Definir el rango de los conjuntos difusos
x_score = np.linspace(0, 1, 100)
output_range = np.linspace(0, 10, 100)

# Definir los conjuntos difusos de salida
output_neg = fuzz.trimf(output_range, [0, 0, 5])
output_neu = fuzz.trimf(output_range, [0, 5, 10])
output_pos = fuzz.trimf(output_range, [5, 10, 10])

# Evaluar las reglas y calcular la fuerza de activación
def evaluate_rules(pos_score, neg_score, pos_low, pos_med, pos_high, neg_low, neg_med, neg_high):
    pos_membership = {
        'Low': fuzz.interp_membership(x_score, pos_low, pos_score),
        'Medium': fuzz.interp_membership(x_score, pos_med, pos_score),
        'High': fuzz.interp_membership(x_score, pos_high, pos_score)
    }
    neg_membership = {
        'Low': fuzz.interp_membership(x_score, neg_low, neg_score),
        'Medium': fuzz.interp_membership(x_score, neg_med, neg_score),
        'High': fuzz.interp_membership(x_score, neg_high, neg_score)
    }

    rule_strengths = {
        'R1': min(pos_membership['Low'], neg_membership['Low']),
        'R2': min(pos_membership['Medium'], neg_membership['Low']),
        'R3': min(pos_membership['High'], neg_membership['Low']),
        'R4': min(pos_membership['Low'], neg_membership['Medium']),
        'R5': min(pos_membership['Medium'], neg_membership['Medium']),
        'R6': min(pos_membership['High'], neg_membership['Medium']),
        'R7': min(pos_membership['Low'], neg_membership['High']),
        'R8': min(pos_membership['Medium'], neg_membership['High']),
        'R9': min(pos_membership['High'], neg_membership['High'])
    }

    w_neg = max(rule_strengths['R4'], rule_strengths['R7'], rule_strengths['R8'])
    w_neu = max(rule_strengths['R1'], rule_strengths['R5'], rule_strengths['R9'])
    w_pos = max(rule_strengths['R2'], rule_strengths['R3'], rule_strengths['R6'])

    return w_neg, w_neu, w_pos

# Agregar las salidas de las reglas
def aggregate_outputs(w_neg, w_neu, w_pos):
    activation_neg = np.fmin(w_neg, output_neg)
    activation_neu = np.fmin(w_neu, output_neu)
    activation_pos = np.fmin(w_pos, output_pos)
    aggregated = np.fmax(activation_neg, np.fmax(activation_neu, activation_pos))
    return aggregated

# Defuzzificación
def defuzzify(aggregated_output):
    
    inicio = time.perf_counter()
    
    sentiment_score = fuzz.defuzz(output_range, aggregated_output, 'centroid')
    
        # Clasificar el puntaje en una clase de sentimiento
    if sentiment_score < 3.3:
        sentiment_class = 'Negative'
    elif sentiment_score < 6.7:
        sentiment_class = 'Neutral'
    else:
        sentiment_class = 'Positive'
    
    
    fin = time.perf_counter()
    
    tiempo_defuzzificacion = (fin - inicio) * 1000  # Tiempo en milisegundos
    
    return sentiment_class, tiempo_defuzzificacion



# Aplicar el sistema difuso a cada registro y almacenar el tiempo de defuzzificación
df[['sentimiento_defuzzificado', 'tiempo_defuzzificacion']] = df.apply(
    lambda row: pd.Series(defuzzify(
        aggregate_outputs(*evaluate_rules(
            row['positive_score'], row['negative_score'], 
            fuzz.trimf(x_score, fuzzy_sets['positive']['Low']),
            fuzz.trimf(x_score, fuzzy_sets['positive']['Medium']),
            fuzz.trimf(x_score, fuzzy_sets['positive']['High']),
            fuzz.trimf(x_score, fuzzy_sets['negative']['Low']),
            fuzz.trimf(x_score, fuzzy_sets['negative']['Medium']),
            fuzz.trimf(x_score, fuzzy_sets['negative']['High'])
        ))
    )), axis=1
)


df['tiempo_total']=df.apply(lambda row: row['tiempo_fuzzification']+row['tiempo_defuzzificacion'],axis=1)


# Guardar el resultado en un nuevo archivo CSV
df.to_csv('resultado.csv', index=False)


# Contar el total de tweets procesados
total_tweets = len(df)

# Contar los tweets por sentimiento defuzzificado
total_positivos = df['sentimiento_defuzzificado'].value_counts().get('Positive', 0)
total_negativos = df['sentimiento_defuzzificado'].value_counts().get('Negative', 0)
total_neutrales = df['sentimiento_defuzzificado'].value_counts().get('Neutral', 0)

# Calcular el tiempo promedio de ejecución
tiempo_promedio_ejecucion = df['tiempo_total'].mean()

# Imprimir los resultados en consola
print(f"Total de tweets procesados: {total_tweets}")
print(f"Total positivos: {total_positivos}")
print(f"Total negativos: {total_negativos}")
print(f"Total neutrales: {total_neutrales}")
print(f"Tiempo promedio de ejecución: {tiempo_promedio_ejecucion:.2f} ms")


print("El archivo csv ha sido procesado con exito!")
