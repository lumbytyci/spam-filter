import data_loader as dl
import prepare
import model

from keras.callbacks import ModelCheckpoint


def train_model(seq_model, config: dict, X_train, y_train, X_test, y_test, save=True):

    checkpoint_file_path = "../checkpoints/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        checkpoint_file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    callbacks = []
    if save:
        callbacks.append(checkpoint)

    seq_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  batch_size=config['train']['batch_size'],
                  epochs=config['train']['epochs'],
                  callbacks=callbacks,
                  verbose=1)

    performance = seq_model.evaluate(X_test, y_test)
    print(performance)
