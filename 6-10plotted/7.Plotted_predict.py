"""
CNN model for predicting labels from EEM images.
Uses plotted images as input (560x480 pixels, RGB).
Runs 10 iterations with different random splits for robust evaluation.
"""
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# Load data - update this path to your data file
new_data = pd.read_excel("./image_path_label.xlsx")

image_paths = new_data["image_path"].values
labels = new_data["label"].values

# Load images
images = []
for image_path in image_paths:
    image = load_img(image_path, target_size=(560, 480))
    image = img_to_array(image)
    images.append(image)
images = np.array(images)

for i in range(10):
    random_state = np.random.randint(1, 200)
    print(f"Running iteration with random_state:{random_state}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=0.15, random_state=random_state)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=random_state)  # 0.2 * 0.8 = 0.16

    # Normalize images
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.

    # Normalize labels
    scaler_labels = MinMaxScaler()
    y_train = scaler_labels.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_labels.transform(y_val.reshape(-1, 1))
    y_test = scaler_labels.transform(y_test.reshape(-1, 1))
    dump(scaler_labels, f'./scaler_labels_{random_state}.joblib')

    # Build model
    input_image = tf.keras.Input(shape=(560, 480, 3))

    x1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = layers.Flatten()(x1)

    x2 = layers.Dense(64, activation='relu')(x1)
    x2 = layers.Dense(64, activation='relu')(x2)
    output = layers.Dense(1)(x2)

    model = Model(inputs=input_image, outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Set early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # Train model
    history =  model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Plot training curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Curve')
    plt.legend()
    plt.savefig(f'training_history_{random_state}.png')
    plt.close()

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)

    # Inverse normalization
    y_train_original = scaler_labels.inverse_transform(y_train)
    y_test_original =scaler_labels.inverse_transform(y_test)
    y_val_original = scaler_labels.inverse_transform(y_val)
    y_pred_train_original = scaler_labels.inverse_transform(y_pred_train)
    y_pred_test_original = scaler_labels.inverse_transform(y_pred_test)
    y_pred_val_original = scaler_labels.inverse_transform(y_pred_val)

    # Save predictions
    data_train = {'Train_Predictions': y_pred_train_original.flatten(),
                  'Train_Actual': y_train_original.flatten()}
    data_test = {'Test_Predictions': y_pred_test_original.flatten(),
                 'Test_Actual': y_test_original.flatten()}
    data_val = {'Val_Predictions': y_pred_val_original.flatten(),
                 'Val_Actual': y_val_original.flatten()}

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)
    df_val = pd.DataFrame(data_val)

    with pd.ExcelWriter(f'actual_predicted_{random_state}.xlsx') as writer:
        df_train.to_excel(writer, sheet_name='Train_Data', index=False)
        df_test.to_excel(writer, sheet_name='Test_Data', index=False)
        df_val.to_excel(writer, sheet_name='Val_Data', index=False)

        # Calculate metrics
    metrics = {
        'train': {
            'mse': mean_squared_error(y_train_original, y_pred_train_original),
            'mae': mean_absolute_error(y_train_original, y_pred_train_original),
            'rmse': np.sqrt(mean_squared_error(y_train_original, y_pred_train_original)),
            'r2': r2_score(y_train_original, y_pred_train_original)
        },
        'test': {
            'mse': mean_squared_error(y_test_original, y_pred_test_original),
            'mae': mean_absolute_error(y_test_original, y_pred_test_original),
            'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred_test_original)),
            'r2': r2_score(y_test_original, y_pred_test_original)
        },
        'val': {
            'mse': mean_squared_error(y_val_original, y_pred_val_original),
            'mae': mean_absolute_error(y_val_original, y_pred_val_original),
            'rmse': np.sqrt(mean_squared_error(y_val_original, y_pred_val_original)),
            'r2': r2_score(y_val_original, y_pred_val_original)
        }
    }

    # Print metrics
    print("\nTrain Metrics:")
    print(f"MSE: {metrics['train']['mse']:.4f}, MAE: {metrics['train']['mae']:.4f}, "
          f"RMSE: {metrics['train']['rmse']:.4f}, R²: {metrics['train']['r2']:.4f}")

    print("Test Metrics:")
    print(f"MSE: {metrics['test']['mse']:.4f}, MAE: {metrics['test']['mae']:.4f}, "
          f"RMSE: {metrics['test']['rmse']:.4f}, R²: {metrics['test']['r2']:.4f}")

    # Save metrics
    metrics_data = {
        'Dataset': ['train'] * 4 + ['validation'] * 4 + ['test'] * 4,
        'Metric': ['MSE', 'MAE', 'RMSE', 'R²'] * 3,
        'Value': [
            metrics['train']['mse'],
            metrics['train']['mae'],
            metrics['train']['rmse'],
            metrics['train']['r2'],
            metrics['val']['mse'],
            metrics['val']['mae'],
            metrics['val']['rmse'],
            metrics['val']['r2'],
            metrics['test']['mse'],
            metrics['test']['mae'],
            metrics['test']['rmse'],
            metrics['test']['r2']
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_excel(f'metrics_{random_state}.xlsx', index=False)

    # Save model
    model.save(f'plotted_RCPM_{random_state}.h5')
    pd.DataFrame(history.history).to_csv(f'training_history_{random_state}.csv')
    tf.keras.backend.clear_session()
    del model

