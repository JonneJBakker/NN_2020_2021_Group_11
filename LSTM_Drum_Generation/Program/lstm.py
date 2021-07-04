""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import matplotlib.pyplot as plt
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


def train_network():
    """ Train a Neural Network to generate drums """
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_drums directory """
    notes = []

    for file in glob.glob("midi_drums/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        current_offset = 0
        for element in notes_to_parse:
            prev_offset = current_offset
            current_offset = element.offset
            offset = current_offset - prev_offset

            if 0 < offset < 0.375:
                offset = 0.25
            elif 0.375 <= offset < 0.625:
                offset = 0.5
            elif 0.625 <= offset < 0.875:
                offset = 0.75
            elif offset >= 0.875:
                offset = 1
            else:
                offset = 0

            if offset != 0:
                for i in range(int((offset-0.25) / 0.25)):
                    notes.append('rest')
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    """ create the structure of the neural network.
        The different network structures can be created by changing
        the n_nodes variable. To create the different network depths:
        Remove all '#' signs in the code below for the network created
        by Skuli. Remove only the single '#' to create Skuli's
        network without the dense layer. """
    n_nodes = 512
    model = Sequential()
    model.add(LSTM(
        n_nodes,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    #model.add(LSTM(n_nodes, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(n_nodes))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    ##model.add(Dense(0.5 * n_nodes))
    ##model.add(Activation('relu'))
    ##model.add(BatchNorm())
    ##model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights_64_Hidden1/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    history = model.fit(network_input, network_output, validation_split=0.2, epochs=200, batch_size=128, callbacks=callbacks_list)
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    train_network()
