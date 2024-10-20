import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet import Prophet
import plotly.graph_objects as go

try:
    # Charger le fichier CSV local
    file_path = 'btc.csv'
    data = pd.read_csv(file_path, delimiter=',')

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

    # Trier les données par date ascendante
    data = data.sort_values('date').reset_index(drop=True)

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

    # Calcul des rendements quotidiens
    data['return'] = data['priceClose'].pct_change()

    # Calcul de la volatilité sur 7 jours basée sur les rendements
    data['volatility_7d'] = data['return'].rolling(window=7).std()

    # Préparation des données supplémentaires
    data['Previous_Price'] = data['priceClose'].shift(1)
    data['SMA_7'] = data['priceClose'].rolling(window=7).mean()
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # Supprimer les lignes contenant des valeurs NaN
    data = data.dropna()

    # Afficher les années minimales et maximales
    min_year = data['year'].min()
    max_year = data['year'].max()
    print(f"Année minimale: {min_year}")
    print(f"Année maximale: {max_year}")

    # Calculer le nombre d'années couvertes
    num_years = max_year - min_year
    print(f"Nombre d'années couvertes: {num_years}")

    if num_years <= 0:
        print("Erreur dans le calcul du taux de croissance : nombre d'années invalide.")
        print("Vérifiez les données de date pour vous assurer qu'elles couvrent plusieurs années.")
        exit()

    # Calculer un taux de croissance annuel moyen basé sur les données historiques
    growth_rate = (data['priceClose'].iloc[-1] / data['priceClose'].iloc[0]) ** (1 / num_years) - 1
    print(f"Taux de croissance annuel moyen: {growth_rate}")

    # Limiter le taux de croissance annuel à 30%
    growth_rate = min(growth_rate, 0.30)
    print(f"Taux de croissance annuel moyen ajusté: {growth_rate}")

    # Sélectionner les caractéristiques et la variable cible
    X = data[
        ['Previous_Price', 'priceOpen', 'priceHigh', 'priceLow', 'volume', 'SMA_7', 'volatility_7d', 'month', 'year']]
    y = data['priceClose']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Créer et entraîner le modèle de Régression Linéaire
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    print("Modèle de régression linéaire entraîné.")

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)
    print("Prédiction effectuée sur l'ensemble de test.")

    # Évaluer les performances du modèle
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (LR): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score (LR): {r2}')

    # Visualisation des prédictions par rapport aux valeurs réelles
    # Vous pouvez décommenter les lignes ci-dessous pour afficher les graphiques
    """
    print("Génération du graphique de régression linéaire...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Prédictions')
    plt.xlabel("Prix Réels")
    plt.ylabel("Prix Prédits")
    plt.title("Régression Linéaire : Prédiction vs Prix Réels")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Référence y=x')
    plt.legend()
    plt.show()
    """

    # Utiliser Prophet pour les prévisions futures jusqu'en 2030
    print("Préparation des données pour Prophet...")
    data_prophet = data[['date', 'priceClose']].rename(columns={'date': 'ds', 'priceClose': 'y'})

    # Initialiser le modèle Prophet avec des paramètres ajustés
    print("Initialisation du modèle Prophet...")
    model_prophet = Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,  # Ajuster selon les besoins
        seasonality_prior_scale=10.0  # Ajuster selon les besoins
    )

    # Ajouter une saisonnalité personnalisée pour les cycles de 4 ans
    print("Ajout de la saisonnalité personnalisée pour les cycles de 4 ans...")
    model_prophet.add_seasonality(name='quadrennial', period=365 * 4, fourier_order=5)

    # Ajuster le modèle
    print("Entraînement du modèle Prophet...")
    model_prophet.fit(data_prophet)
    print("Modèle Prophet entraîné.")

    # Créer un dataframe pour les dates futures (jusqu'à 2030)
    print("Création des dates futures pour les prévisions...")
    future = model_prophet.make_future_dataframe(periods=365 * 6, freq='D')  # 6 ans supplémentaires
    print("Dates futures créées.")

    # Prédiction avec Prophet
    print("Prédiction avec Prophet...")
    forecast = model_prophet.predict(future)
    print("Prédiction Prophet effectuée.")

    # **Étape supplémentaire : Analyser les cycles historiques**

    # Ajouter une colonne pour le log du prix pour analyser les variations logarithmiques
    data['log_price'] = np.log(data['priceClose'])

    # Calculer la variation quotidienne du prix
    data['price_change'] = data['log_price'].diff()

    # Définir un seuil pour déterminer les phases de marché
    threshold = data['price_change'].std() * 2  # Vous pouvez ajuster ce multiplicateur

    # Initialiser la colonne 'market_phase'
    data['market_phase'] = 'Stable'
    data.loc[data['price_change'] > threshold, 'market_phase'] = 'Bull'
    data.loc[data['price_change'] < -threshold, 'market_phase'] = 'Bear'

    # Identifier les transitions entre les phases
    data['phase_change'] = data['market_phase'] != data['market_phase'].shift(1)

    # Regrouper les phases
    phases = []
    current_phase = {'phase': data['market_phase'].iloc[0], 'start_date': data['date'].iloc[0]}
    for idx, row in data.iterrows():
        if row['phase_change']:
            current_phase['end_date'] = data['date'].iloc[idx - 1]
            phases.append(current_phase)
            current_phase = {'phase': row['market_phase'], 'start_date': row['date']}
    current_phase['end_date'] = data['date'].iloc[-1]
    phases.append(current_phase)

    # Convertir en DataFrame
    phases_df = pd.DataFrame(phases)
    phases_df['duration'] = (phases_df['end_date'] - phases_df['start_date']).dt.days

    # Ajouter les prix de début et de fin pour chaque phase
    phases_df['start_price'] = phases_df['start_date'].apply(lambda x: data.loc[data['date'] == x, 'priceClose'].values[0])
    phases_df['end_price'] = phases_df['end_date'].apply(lambda x: data.loc[data['date'] == x, 'priceClose'].values[0])
    phases_df['price_change'] = (phases_df['end_price'] - phases_df['start_price']) / phases_df['start_price']

    # Calculer les statistiques moyennes
    bull_phases = phases_df[phases_df['phase'] == 'Bull']
    bear_phases = phases_df[phases_df['phase'] == 'Bear']

    avg_bull_duration = bull_phases['duration'].mean()
    avg_bull_return = bull_phases['price_change'].mean()

    avg_bear_duration = bear_phases['duration'].mean()
    avg_bear_return = bear_phases['price_change'].mean()

    print(f"Durée moyenne des Bull Markets: {avg_bull_duration} jours")
    print(f"Retour moyen des Bull Markets: {avg_bull_return * 100:.2f}%")
    print(f"Durée moyenne des Bear Markets: {avg_bear_duration} jours")
    print(f"Retour moyen des Bear Markets: {avg_bear_return * 100:.2f}%")

    # Définir la fonction de simulation basée sur les cycles historiques
    def simulate_price_with_historical_cycles(last_price, total_days, avg_bull_duration, avg_bull_return, avg_bear_duration, avg_bear_return, volatility):
        """
        Simule les prix en utilisant les durées et rendements moyens des cycles historiques.

        :param last_price: Prix initial pour la simulation.
        :param total_days: Nombre total de jours à simuler.
        :param avg_bull_duration: Durée moyenne des bull markets.
        :param avg_bull_return: Retour moyen des bull markets.
        :param avg_bear_duration: Durée moyenne des bear markets.
        :param avg_bear_return: Retour moyen des bear markets.
        :param volatility: Volatilité basée sur les rendements quotidiens.
        :return: Liste des prix simulés.
        """
        print("Début de la simulation des prix avec cycles historiques...")
        prices = []
        current_price = last_price
        current_phase = 'Bull'  # Commencer par un bull market
        days_in_phase = 0

        if np.isnan(avg_bull_duration) or np.isnan(avg_bear_duration) or avg_bull_duration <= 0 or avg_bear_duration <= 0:
            print("Erreur : Les durées moyennes des cycles ne sont pas disponibles ou invalides.")
            return [last_price] * total_days

        phase_duration = avg_bull_duration
        phase_return = avg_bull_return

        for day in range(total_days):
            if days_in_phase >= phase_duration:
                # Changer de phase
                if current_phase == 'Bull':
                    current_phase = 'Bear'
                    phase_duration = avg_bear_duration
                    phase_return = avg_bear_return
                else:
                    current_phase = 'Bull'
                    phase_duration = avg_bull_duration
                    phase_return = avg_bull_return
                days_in_phase = 0

            if phase_duration <= 0:
                daily_return = 0
            else:
                daily_return = (1 + phase_return) ** (1 / phase_duration) - 1

            # Appliquer le rendement quotidien et la volatilité
            shock = np.random.normal(0, volatility)
            new_price = current_price * (1 + daily_return + shock)
            new_price = max(new_price, 1)
            prices.append(new_price)

            current_price = new_price
            days_in_phase += 1

            if (day + 1) % 500 == 0:
                print(f"Simulation en cours... Jour {day + 1} sur {total_days}")

        print("Simulation des prix terminée.")
        return prices

    # Calcul de la volatilité moyenne basée sur les rendements
    volatility = data['volatility_7d'].mean()
    print(f"Volatilité moyenne sur 7 jours (rendements): {volatility}")

    # Calcul du nombre total de jours à simuler
    total_days = len(future)
    print(f"Nombre total de jours à simuler : {total_days}")

    # Simuler le prix du Bitcoin jusqu'en 2030 à partir du dernier point connu
    print("Début de la simulation des prix futurs avec cycles historiques...")
    simulated_prices_historical = simulate_price_with_historical_cycles(
        last_price=data['priceClose'].iloc[-1],
        total_days=total_days,
        avg_bull_duration=avg_bull_duration,
        avg_bull_return=avg_bull_return,
        avg_bear_duration=avg_bear_duration,
        avg_bear_return=avg_bear_return,
        volatility=volatility
    )
    print("Simulation des prix futurs terminée.")

    # Vérifier les longueurs
    forecast_length = len(forecast)
    simulated_length = len(simulated_prices_historical)
    print(f"Longueur du dataframe forecast : {forecast_length}")
    print(f"Longueur de la liste simulated_prices_historical : {simulated_length}")

    # Ajuster la liste des prix simulés si nécessaire
    if simulated_length < forecast_length:
        last_price_sim = simulated_prices_historical[-1]
        additional_days = forecast_length - simulated_length
        simulated_prices_historical += [last_price_sim] * additional_days
        print(f"Liste des prix simulés complétée avec {additional_days} valeurs.")
    elif simulated_length > forecast_length:
        simulated_prices_historical = simulated_prices_historical[:forecast_length]
        print(f"Liste des prix simulés tronquée à {forecast_length} valeurs.")

    # Ajouter les prix simulés au dataframe de prévision
    forecast['simulated_price_historical'] = simulated_prices_historical
    print("Prix simulés ajoutés au dataframe forecast.")

    # Visualisation de style Binance avec Plotly
    print("Création du graphique interactif avec Plotly...")
    fig = go.Figure()

    # Ajouter les prix historiques
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['priceClose'],
        name='Prix Historique',
        line=dict(color='blue')
    ))

    # Ajouter les prévisions Prophet
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Prévision Prophet',
        line=dict(color='red', dash='dot')
    ))

    # Ajouter les limites de confiance Prophet
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        name='Limite Supérieure Prophet',
        line=dict(color='green', dash='dot'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        name='Limite Inférieure Prophet',
        line=dict(color='orange', dash='dot'),
        fill='tonexty',
        showlegend=False
    ))

    # Ajouter les prix simulés basés sur les cycles historiques
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['simulated_price_historical'],
        name='Prix Simulé (Cycles Historiques)',
        line=dict(color='purple')
    ))

    # Configurer le layout du graphique
    fig.update_layout(
        title="Prévision du Prix du Bitcoin jusqu'en 2030",
        xaxis_title="Date",
        yaxis_title="Prix (USD)",
        template="plotly_dark",
        legend=dict(x=0, y=1)
    )

    fig.show()
    print("Graphique interactif affiché.")

except Exception as e:
    print(f"Une erreur s'est produite: {e}")
    exit()
