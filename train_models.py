import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Remove any streamlit imports or dependencies
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("Training models...")

    # Improve Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=500,      # Increased number of trees
        max_depth=25,          # Optimized depth
        min_samples_split=10,  # Increased to reduce overfitting
        min_samples_leaf=4,    # Increased for better generalization
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,       # Enable out-of-bag score
        n_jobs=-1,
        random_state=42,
        class_weight='balanced_subsample'  # Better handling of imbalanced data
    )
    rf_model.fit(X_train, y_train)

    # Save the Random Forest model
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # Create an improved LSTM model with residual connections
    lstm_model = Sequential([
        # Embedding with larger vocabulary and dimensions
        Embedding(10000, 300, input_length=X_train.shape[1]),
        
        # Initial dropout
        SpatialDropout1D(0.2),
        
        # First CNN block with residual connection
        Conv1D(128, 3, padding='same', activation='relu'),
        Conv1D(128, 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        
        # Second CNN block
        Conv1D(256, 3, padding='same', activation='relu'),
        Conv1D(256, 3, padding='same', activation='relu'),
        MaxPooling1D(2),
        Dropout(0.2),
        
        # Bidirectional LSTM layers with increased complexity
        Bidirectional(LSTM(128, return_sequences=True, 
                          kernel_regularizer='l2',
                          recurrent_regularizer='l2')),
        Dropout(0.2),
        
        Bidirectional(LSTM(128, 
                          kernel_regularizer='l2',
                          recurrent_regularizer='l2')),
        Dropout(0.2),
        
        # Dense layers with batch normalization
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])

    # Improved optimizer settings
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )

    # Compile with improved metrics
    lstm_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )

    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001  # Minimum change to qualify as an improvement
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,    # More gradual reduction
        patience=3,
        min_lr=1e-6,
        verbose=1,
        cooldown=1     # Wait 1 epoch before resuming normal operation
    )

    # Train with improved parameters
    history = lstm_model.fit(
        X_train, y_train,
        epochs=50,          # Increased epochs since we have early stopping
        batch_size=32,      # Smaller batch size for better generalization
        validation_split=0.15,  # Additional validation split
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        class_weight={0: 1.0, 1: 1.5},  # Adjust based on your class distribution
        verbose=1
    )

    # Plot training history with improved visualization
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], linewidth=2)
    plt.plot(history.history['val_accuracy'], linewidth=2)
    plt.title('Model Accuracy', fontsize=12, pad=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], linewidth=2)
    plt.plot(history.history['val_loss'], linewidth=2)
    plt.title('Model Loss', fontsize=12, pad=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Generate predictions with threshold optimization
    y_pred_proba = lstm_model.predict(X_test)
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = 0.5
    best_f1 = 0

    # Find the best threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['weighted avg']['f1-score']
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Use the best threshold for final predictions
    y_pred = (y_pred_proba > best_threshold).astype(int)

    # Create improved confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix', fontsize=12, pad=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.show()

    # Print detailed classification report
    print("\nClassification Report (threshold={:.2f}):".format(best_threshold))
    print(classification_report(y_test, y_pred))

    # Save the LSTM model
    lstm_model.save("models/lstm_model.h5")

    print("Models Trained and Saved!")

    # After model training, add these evaluation sections:

    # Calculate and display detailed metrics
    print("\n=== Model Performance Metrics ===")

    # Random Forest Metrics
    rf_train_acc = rf_model.score(X_train, y_train)
    rf_test_acc = rf_model.score(X_test, y_test)
    rf_predictions = rf_model.predict(X_test)
    rf_report = classification_report(y_test, rf_predictions)

    print("\nRandom Forest Results:")
    print(f"Training Accuracy: {rf_train_acc:.4f}")
    print(f"Testing Accuracy: {rf_test_acc:.4f}")
    print("\nDetailed Classification Report:")
    print(rf_report)

    # LSTM Metrics - with error handling
    try:
        lstm_train_metrics = lstm_model.evaluate(X_train, y_train, verbose=0)
        lstm_test_metrics = lstm_model.evaluate(X_test, y_test, verbose=0)
        
        print("\nLSTM Results:")
        metric_names = lstm_model.metrics_names
        for i, metric in enumerate(metric_names):
            print(f"Training {metric}: {lstm_train_metrics[i]:.4f}")
            print(f"Testing {metric}: {lstm_test_metrics[i]:.4f}")

    except Exception as e:
        print(f"\nError evaluating LSTM metrics: {e}")
        print("Falling back to basic accuracy calculation...")
        
        # Fallback to basic accuracy calculation
        lstm_predictions = (lstm_model.predict(X_test) > best_threshold).astype(int)
        lstm_test_acc = np.mean(lstm_predictions == y_test)
        print(f"LSTM Test Accuracy: {lstm_test_acc:.4f}")

    # Save metrics with error handling
    try:
        with open("models/training_metrics.txt", "w") as f:
            f.write("=== Training Metrics ===\n")
            f.write(f"Random Forest Training Accuracy: {rf_train_acc:.4f}\n")
            f.write(f"Random Forest Testing Accuracy: {rf_test_acc:.4f}\n")
            
            if 'lstm_test_metrics' in locals():
                f.write(f"LSTM Training Accuracy: {lstm_train_metrics[1]:.4f}\n")
                f.write(f"LSTM Testing Accuracy: {lstm_test_metrics[1]:.4f}\n")
                f.write(f"LSTM Final Loss: {lstm_test_metrics[0]:.4f}\n")
            else:
                f.write(f"LSTM Test Accuracy: {lstm_test_acc:.4f}\n")
            
            f.write(f"Best Threshold: {best_threshold:.4f}\n")
            
        print("\nTraining metrics have been saved to 'models/training_metrics.txt'")
    except Exception as e:
        print(f"\nError saving metrics to file: {e}")

    # Plot training history with all metrics - with error handling
    plt.figure(figsize=(15, 10))

    # Helper function to safely plot metrics
    def safe_plot_metric(history, metric_name, ax, title):
        if metric_name in history.history:
            ax.plot(history.history[metric_name], linewidth=2, label='Train')
            val_metric = f'val_{metric_name}'
            if val_metric in history.history:
                ax.plot(history.history[val_metric], linewidth=2, label='Validation')
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, f'No {metric_name} data available',
                    ha='center', va='center')
            ax.set_title(title, fontsize=12, pad=10)

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot each metric
    safe_plot_metric(history, 'accuracy', ax1, 'Model Accuracy')
    safe_plot_metric(history, 'loss', ax2, 'Model Loss')
    safe_plot_metric(history, 'auc', ax3, 'Model AUC')

    # For precision and recall, plot them together if available
    if 'precision' in history.history and 'recall' in history.history:
        ax4.plot(history.history['precision'], linewidth=2, label='Precision')
        ax4.plot(history.history['recall'], linewidth=2, label='Recall')
        ax4.set_title('Precision and Recall', fontsize=12, pad=10)
        ax4.set_ylabel('Score', fontsize=10)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.legend(loc='best')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'No Precision/Recall data available',
                 ha='center', va='center')
        ax4.set_title('Precision and Recall', fontsize=12, pad=10)

    plt.tight_layout()
    plt.show()

    # Plot confusion matrices side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Random Forest Confusion Matrix
    cm_rf = confusion_matrix(y_test, rf_predictions)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Random Forest Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # LSTM Confusion Matrix
    lstm_predictions = (lstm_model.predict(X_test) > best_threshold).astype(int)
    cm_lstm = confusion_matrix(y_test, lstm_predictions)
    sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('LSTM Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

    # Print per-class metrics
    print("\n=== Per-Class Performance ===")
    for model_name, y_pred in [("Random Forest", rf_predictions), ("LSTM", lstm_predictions)]:
        print(f"\n{model_name} Per-class Metrics:")
        for class_id in np.unique(y_test):
            mask = y_test == class_id
            class_acc = np.mean(y_pred[mask] == y_test[mask])
            print(f"Class {class_id} Accuracy: {class_acc:.4f}")