import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet import Prophet
import plotly.graph_objects as go
import datetime

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

    # Préparation des données supplémentaires
    data['Previous_Price'] = data['priceClose'].shift(1)
    data['SMA_7'] = data['priceClose'].rolling(window=7).mean()
    data['volatility_7d'] = data['priceClose'].rolling(window=7).std()
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
    print("Génération du graphique de régression linéaire...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Prédictions')
    plt.xlabel("Prix Réels")
    plt.ylabel("Prix Prédits")
    plt.title("Régression Linéaire : Prédiction vs Prix Réels")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Référence y=x')

    # Option 1 : Affichage non bloquant
    plt.legend()
    plt.show(block=False)
    plt.pause(5)  # Afficher le graphique pendant 5 secondes
    plt.close()
    print("Graphique de régression linéaire affiché et fermé automatiquement.")

    # Option 2 : Sauvegarder le graphique
    # print("Génération et sauvegarde du graphique de régression linéaire...")
    # plt.savefig('regression_plot.png')  # Sauvegarder le graphique
    # plt.close()  # Fermer le graphique
    # print("Graphique de régression linéaire sauvegardé sous 'regression_plot.png'.")

    # Tracer les séries temporelles des valeurs réelles et prédites
    print("Génération du graphique des séries temporelles...")
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.values, label='Prix Réel')
    plt.plot(y_pred, label='Prix Prédit')
    plt.xlabel("Index")
    plt.ylabel("Prix Close")
    plt.title("Prix Réel vs. Prix Prédit")
    plt.legend()
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    print("Graphique des séries temporelles affiché et fermé automatiquement.")

    # Tracer la distribution des résidus
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7, color='purple')
    plt.xlabel("Résidus")
    plt.ylabel("Fréquence")
    plt.title("Distribution des Résidus")
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    print("Distribution des résidus affichée et fermée automatiquement.")

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


    # Définir la fonction de simulation révisée
    def simulate_price_with_cycles_v6(last_price, annual_growth_rate, volatility, days):
        print("Début de la simulation des prix avec cycles...")
        prices = [last_price]
        cycle_length = 365 * 4  # 4 ans
        bull_phase = int(cycle_length * 0.4)  # 40% du cycle (~584 jours)
        bear_phase = int(cycle_length * 0.3)  # 30% du cycle (~438 jours)
        stable_phase = cycle_length - bull_phase - bear_phase  # ~438 jours

        for day in range(1, days):
            cycle_day = day % cycle_length
            if cycle_day < bull_phase:
                # Bull phase: croissance annuelle répartie sur la phase
                daily_growth = (1 + annual_growth_rate) ** (1 / cycle_length) - 1
                daily_growth *= 1.1  # Augmenter légèrement la croissance pendant le bull
            elif cycle_day < bull_phase + bear_phase:
                # Bear phase: décroissance annuelle
                daily_growth = (1 - annual_growth_rate) ** (1 / cycle_length) - 1
                daily_growth *= 1.1  # Accentuer la décroissance pendant le bear
            else:
                # Stable phase: légère fluctuation autour de la moyenne
                daily_growth = 0

            # Appliquer une volatilité
            shock = np.random.normal(0, volatility / 100)
            new_price = prices[-1] * (1 + daily_growth + shock)
            prices.append(max(new_price, 0))  # Le prix ne peut pas être négatif

            # Afficher un message tous les 500 jours pour suivre la progression
            if day % 500 == 0:
                print(f"Simulation en cours... Jour {day} sur {days}")

        print("Simulation des prix terminée.")
        return prices


    # Simuler le prix du Bitcoin jusqu'en 2030 à partir du dernier point connu
    print("Début de la simulation des prix futurs...")
    simulated_prices_v6 = simulate_price_with_cycles_v6(
        last_price=data['priceClose'].iloc[-1],
        annual_growth_rate=growth_rate,
        volatility=data['volatility_7d'].mean(),
        days=len(future)
    )
    print("Simulation des prix futurs terminée.")

    # Ajouter les prix simulés au dataframe de prévision
    print("Ajout des prix simulés au dataframe de prévision...")
    forecast['simulated_price_v6'] = simulated_prices_v6
    print("Prix simulés ajoutés.")

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

    # Ajouter les prix simulés basés sur la projection des cycles améliorés
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['simulated_price_v6'],
        name='Prix Simulé (Cycles 4 ans)',
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

    # Option 1 : Affichage dans un environnement interactif (non bloquant)
    fig.show()

    # Option 2 : Sauvegarder le graphique interactif
    # fig.write_html('forecast_plot.html')
    # print("Graphique interactif sauvegardé sous 'forecast_plot.html'.")

    print("Graphique interactif affiché.")

except Exception as e:
    print(f"Une erreur s'est produite: {e}")
    exit()



