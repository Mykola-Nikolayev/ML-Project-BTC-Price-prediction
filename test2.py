import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
import plotly.graph_objects as go
import datetime

# Charger le fichier CSV local
file_path = 'btc.csv'
data = pd.read_csv(file_path, delimiter=',')

# Afficher les colonnes pour vérifier les noms exacts (pour débogage)
print(data.columns)

# Supprimer les espaces avant/après dans les noms de colonnes
data.columns = data.columns.str.strip()

# Renommer les colonnes pour simplifier la manipulation
data.rename(columns={
    'Date': 'date',
    'Dernier': 'priceClose',
    'Ouv.': 'priceOpen',
    'Plus Haut': 'priceHigh',
    'Plus Bas': 'priceLow',
    'Vol.': 'volume'
}, inplace=True)

# Convertir la colonne 'date' en format datetime correct pour Prophet
data['date'] = pd.to_datetime(data['date'], dayfirst=True, errors='coerce')

# Supprimer les lignes contenant des dates non valides
data = data.dropna(subset=['date'])

# Convertir les colonnes de prix et volume en format numérique
data['priceClose'] = data['priceClose'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
data['priceOpen'] = data['priceOpen'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
data['priceHigh'] = data['priceHigh'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
data['priceLow'] = data['priceLow'].str.replace('.', '', regex=False).str.replace(',', '.').astype(float)
data['volume'] = (
    data['volume']
    .str.replace('B', 'e9')
    .str.replace('M', 'e6')
    .str.replace('K', 'e3')
    .str.replace('.', '', regex=False)
    .str.replace(',', '.')
    .astype(float)
)

# Préparation des données supplémentaires
# Créer une colonne Previous_Price à partir de Price_Close
data['Previous_Price'] = data['priceClose'].shift(1)

# Calculer la moyenne mobile sur 7 jours
data['SMA_7'] = data['priceClose'].rolling(window=7).mean()

# Ajouter une colonne pour la volatilité sur 7 jours
data['volatility_7d'] = data['priceClose'].rolling(window=7).std()

# Ajouter une colonne indiquant le mois et l'année pour capturer la saisonnalité
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Supprimer les lignes contenant des valeurs NaN (résultant de Previous_Price, SMA_7, ou volatilité)
data = data.dropna()

# Vérifier si data contient suffisamment de lignes après suppression des NaN
if len(data) < 2:
    print("Pas assez de données pour entraîner le modèle.")
    exit()

# Sélectionner les caractéristiques et la variable cible
X = data[['Previous_Price', 'priceOpen', 'priceHigh', 'priceLow', 'volume', 'SMA_7', 'volatility_7d', 'month', 'year']]
y = data['priceClose']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test_scaled)

# Evaluer les performances du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Visualisation des prédictions par rapport aux valeurs réelles
plt.scatter(y_test, y_pred)
plt.xlabel("Prix Reels")
plt.ylabel("Prix Predits")
plt.title("Prediction vs Prix Reels")
plt.show(block=False)
plt.pause(3)
plt.close()

# Utiliser Prophet pour les prévisions futures jusqu'en 2030
data_prophet = data[['date', 'priceClose']].rename(columns={'date': 'ds', 'priceClose': 'y'})

# Initialiser et ajuster le modèle de prévision
model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
model_prophet.fit(data_prophet)

# Créer un dataframe pour les dates futures (jusqu'à 2030)
future = model_prophet.make_future_dataframe(periods=365*6, freq='D')
forecast = model_prophet.predict(future)

# Ajouter une simulation basée sur les cycles historiques du Bitcoin
# Calculer un taux de croissance annuel moyen basé sur les données historiques
growth_rate = (data['priceClose'].iloc[-1] / data['priceClose'].iloc[0]) ** (1 / (data['year'].iloc[-1] - data['year'].iloc[0])) - 1

# Ajouter une simulation de croissance cyclique avec des phases de marché (bull et bear)
def simulate_price_with_cycles_v5(last_price, growth_rate, volatility, days):
    prices = [last_price]
    cycle_length = 365  # Approximation d'un cycle annuel
    bull_phase = int(cycle_length * 0.4)  # 40% du cycle est un bull market
    bear_phase = int(cycle_length * 0.3)  # 30% du cycle est un bear market
    stable_phase = cycle_length - bull_phase - bear_phase  # Le reste est stable

    for day in range(1, days):
        # Déterminer la phase du jour courant
        cycle_day = day % cycle_length
        if cycle_day < bull_phase:
            # Bull phase: forte croissance positive mais limitée
            daily_growth = min(growth_rate / 365 * 0.8, 0.001)  # Limiter la croissance pour éviter des valeurs extrêmes
            growth = np.log(1 + daily_growth)
        elif cycle_day < bull_phase + bear_phase:
            # Bear phase: décroissance
            daily_growth = -min(growth_rate / 365 * 0.8, 0.001)  # Limiter la décroissance
            growth = np.log(1 + daily_growth)
        else:
            # Stable phase: légère fluctuation autour de la moyenne
            daily_growth = growth_rate / 365 * 0.5
            growth = np.log(1 + daily_growth)

        # Ajouter un choc de volatilité modéré
        shock = np.random.normal(0, volatility / 10000)  # Ajuster la volatilité pour éviter des valeurs aberrantes
        new_price = prices[-1] * np.exp(growth + shock)
        prices.append(max(new_price, 0))  # Le prix ne peut pas être négatif
    return prices

# Simuler le prix du Bitcoin jusqu'en 2030 à partir du dernier point connu
simulated_prices_v5 = simulate_price_with_cycles_v5(data['priceClose'].iloc[-1], growth_rate, data['volatility_7d'].mean(), len(future))

# Ajouter les prix simulés au dataframe de prévision
forecast['simulated_price_v5'] = simulated_prices_v5

# Visualisation des prévisions futures
fig = model_prophet.plot(forecast)
plt.title("Prévision du Prix du Bitcoin jusqu'en 2030")
plt.xlabel("Date")
plt.ylabel("Prix Prévu")
plt.show(block=False)
plt.pause(3)
plt.close()

# Visualisation de style Binance avec Plotly
fig = go.Figure()

# Ajouter les prix historiques
fig.add_trace(go.Scatter(x=data['date'], y=data['priceClose'], name='Prix Historique', line=dict(color='blue')))

# Ajouter les prévisions jusqu'en 2030
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prévision', line=dict(color='red', dash='dot')))

# Ajouter des lignes de limite supérieure et inférieure pour montrer la bande de confiance du modèle
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Limite Supérieure', line=dict(color='green', dash='dot')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Limite Inférieure', line=dict(color='orange', dash='dot')))

# Ajouter les prix simulés basés sur la projection des cycles améliorés
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['simulated_price_v5'], name='Prix Simulé', line=dict(color='purple')))

# Configurer le layout du graphique
fig.update_layout(
    title="Prévision du Prix du Bitcoin jusqu'en 2030",
    xaxis_title="Date",
    yaxis_title="Prix (USD)",
    template="plotly_dark"
)

# Afficher le graphique interactif
fig.show()




