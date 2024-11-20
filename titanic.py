import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import sys

sys.stdout.reconfigure(encoding='utf-8')

def extract_title(name):
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare"
    }
    
    title = name.str.extract(' ([A-Za-z]+)\.', expand=False)
    return title.map(title_mapping)

def create_family_features(df):
    df = df.copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    conditions = [
        (df['FamilySize'] == 1),
        (df['FamilySize'] <= 4),
        (df['FamilySize'] <= 7),
        (df['FamilySize'] > 7)
    ]
    choices = ['Solo', 'Pequeña', 'Mediana', 'Grande']
    df['FamilyGroup'] = np.select(conditions, choices, default='Grande')
    
    return df

def load_and_preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path, encoding='utf-8')
    test_data = pd.read_csv(test_path, encoding='utf-8')
    
    # Crear una copia para evitar las advertencias de SettingWithCopyWarning
    combined_data = pd.concat([train_data, test_data], sort=False).copy()
    
    # Imputación de valores faltantes
    combined_data['Age'] = combined_data.groupby(['Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median()))
    combined_data['Fare'] = combined_data.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median()))
    
    # Corregir el uso de inplace
    combined_data.loc[combined_data['Embarked'].isna(), 'Embarked'] = combined_data['Embarked'].mode()[0]
    
    # Extraer títulos y crear características de familia
    combined_data['Title'] = extract_title(combined_data['Name'])
    combined_data = create_family_features(combined_data)
    
    # Crear características adicionales
    combined_data['Sex'] = combined_data['Sex'].map({'male': 1, 'female': 0})
    combined_data['Deck'] = combined_data['Cabin'].str.extract('([A-Z])', expand=False)
    combined_data.loc[combined_data['Deck'].isna(), 'Deck'] = 'U'
    
    combined_data['FarePerPerson'] = combined_data['Fare'] / combined_data['FamilySize']
    
    # Características específicas
    combined_data['IsMother'] = ((combined_data['Sex'] == 0) & 
                                (combined_data['Parch'] > 0) & 
                                (combined_data['Age'] > 18)).astype(int)
    combined_data['IsChild'] = (combined_data['Age'] <= 12).astype(int)
    combined_data['IsElderly'] = (combined_data['Age'] >= 65).astype(int)
    
    # Convertir edad y tarifa en categorías
    combined_data['AgeCat'] = pd.qcut(combined_data['Age'], 5, labels=False)
    combined_data['FareCat'] = pd.qcut(combined_data['Fare'], 5, labels=False)
    
    # Codificación one-hot
    categorical_features = ['Embarked', 'Title', 'Deck', 'FamilyGroup']
    combined_data = pd.get_dummies(combined_data, columns=categorical_features, drop_first=True)
    
    return combined_data

def create_model(input_dim, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

def main():
    print("Cargando y preprocesando datos...")
    combined_data = load_and_preprocess_data('train.csv', 'test.csv')
    
    # Separar datos de entrenamiento y prueba
    train_data = combined_data[combined_data['Survived'].notna()].copy()
    test_data = combined_data[combined_data['Survived'].isna()].copy()
    
    # Crear diccionario de características por nombre
    name_to_features = train_data.set_index('Name').to_dict('index')
    
    # Seleccionar características
    features = [col for col in train_data.columns if col not in 
               ['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin']]
    
    X = train_data[features].values
    y = train_data['Survived'].values
    X_test = test_data[features].values
    
    print("Escalando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Dividir datos
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=0.00001
    )
    
    print(f"Entrenando modelo con {X_train.shape[1]} características...")
    model = create_model(input_dim=X_train.shape[1])
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluar modelo
    print("\nEvaluación del modelo:")
    loss, accuracy, auc = model.evaluate(X_val, y_val, verbose=0)
    print(f'1: ---- {accuracy:.4f}')
    print(f'2: ---- {auc:.4f}')
    
    # Mostrar reporte de clasificación
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print("\nReporte de Clasificación:")
    print(classification_report(y_val, y_pred))
    
    def predict_survival(name):
        if name not in name_to_features:
            return "Pasajero no encontrado en los datos."
        
        passenger = name_to_features[name]
        passenger_features = []
        for feat in features:
            passenger_features.append(passenger[feat])
        
        scaled_features = scaler.transform([passenger_features])
        prediction = model.predict(scaled_features)
        
        probability = prediction[0][0]
        survived = "sobrevivió" if probability > 0.5 else "no sobrevivió"
        
        additional_info = f"""
        Información adicional del pasajero:
        - Clase: {passenger['Pclass']}
        - Edad: {passenger['Age']:.0f}
        - Género: {'Mujer' if passenger['Sex'] == 0 else 'Hombre'}
        - Familia a bordo: {passenger['FamilySize']}
        - Tarifa pagada: {passenger['Fare']:.2f}
        """
        
        return f"El modelo predice que {name} {survived} con una probabilidad de {probability:.2f}\n{additional_info}"

    return model, predict_survival, features

if __name__ == "__main__":
    model, predict_survival, features = main()
    
    # Ejemplo de predicción
    nombre_pasajero = "Bonnell, Miss. Elizabeth"
    resultado = predict_survival(nombre_pasajero)
    print("\nPredicción para un pasajero específico:")
    print(resultado)
    
    # Mostrar métricas finales del modelo
    print("\nCaracterísticas utilizadas en el modelo:")
    print(f"Número total de características: {len(features)}")
    print("Características principales:", ", ".join(features[:10]))