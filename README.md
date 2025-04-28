# LSTM Network Experiments for MIDI Music Generation

## Introduction and Purpose

The purpose of this project is to investigate methods for generating symbolic music (in MIDI format) using Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks. Two different approaches were tested and compared in the project:

1. An LSTM model that only predicts note sequences (pitch sequence).
2. An LSTM model that predicts an event sequence containing both notes and timing information (time shifts) between notes.

The goal is to observe the effect of including rhythm information on the quality of the generated music.

## Dataset

* **Source:** A subset of the [Lakh MIDI Dataset (Clean)](https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean)
  
## Introduction and Purpose

The purpose of this project is to investigate methods for generating symbolic music (in MIDI format) using Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks. Two different approaches were tested and compared in the project:

1. An LSTM model that only predicts note sequences (pitch sequence).
2. An LSTM model that predicts an event sequence containing both notes and timing information (time shifts) between notes.

The goal is to observe the effect of including rhythm information on the quality of the generated music.

## Dataset

* **Source:** A subset of the "Lakh MIDI Dataset (Clean)" dataset found on Kaggle was used.
* **Subset:** Specifically, **43 MIDI files** from the folder containing MIDI files of the `Aerosmith` band were used for training. This was done to enable the model to learn a specific style (rock/pop) and to keep the dataset at a manageable size.

## Method

Two different LSTM-based models were developed and trained in the project:

### Approach 1: Pitch-Only Prediction

1. **Data Preprocessing:** Aerosmith MIDI files were read using the `pretty_midi` library. The note sequence (only pitch values) of the instrument with the most notes in each file was extracted, and notes from all files were combined to create a single long note sequence. A dictionary (vocabulary) was created from the unique notes in this sequence, and the note sequence was converted to integer indices. Input/output pairs (input: 100 notes, output: 101st note) were prepared for training using a sliding window method with fixed length (`sequence_length=100`).
2. **Model Architecture:** A PyTorch model (`MusicLSTM`) consisting of an Embedding layer, a 2-layer LSTM (`hidden_size=512`) followed by a Dropout layer, and a Linear layer producing output in the size of the vocabulary was defined.
3. **Training:** The model was trained on the prepared note sequences to learn to predict the next note using `CrossEntropyLoss` and the `Adam` optimizer.
4. **Generation:** Using the trained model, a new pitch sequence was generated iteratively, starting from a randomly selected seed sequence from the training data. Randomness was controlled by the `temperature` parameter during generation.
5. **Conversion to MIDI:** The generated note sequence was converted to a `.mid` file using `pretty_midi` by **assigning a fixed duration and fixed time interval to each note**. It is known that this approach does not represent rhythmic information.

### Approach 2: Event-Based (Pitch + Time Shift) Prediction

1. **Data Preprocessing:** When reading MIDI files, both note onsets (pitch) and **time differences (time shifts)** between consecutive notes were extracted. Time differences were quantized by dividing them into a certain number (e.g., 32) of discrete categories (quantile bins) according to their distribution. The music sequence was converted into a sequence of events in the form of `NOTE_pitch` and `TIME_binIndex`. A new, larger dictionary including both note and time events was created, and the event sequence was converted to integer indices. Event sequences of fixed length (`sequence_length_event=150`) and output pairs predicting the next event were prepared for training.
2. **Model Architecture:** The **same LSTM architecture** as in Approach 1 was used, but the dimensions of the Embedding and Linear layers were updated according to the size of the new and larger event dictionary (`n_vocab_event`).
3. **Training:** The model was trained on event sequences to learn to predict the next event (whether a note or a time interval) using the same loss function and optimizer.
4. **Generation:** Using the trained model, a new **event** sequence was generated iteratively, starting from a random seed event sequence.
5. **Conversion to MIDI:** The generated event sequence was interpreted:
   * When a `TIME_` event was predicted: The corresponding (average) duration was calculated and added to the current time in the MIDI. This duration was also stored as the potential duration of the next note.
   * When a `NOTE_` event was predicted: The note was placed at the current time, and its duration was assigned as the duration of the `TIME_` event that immediately preceded it (a simple rhythmic assumption).
   * Results were converted to a `.mid` file using `pretty_midi`.

## Results and Comparison

* **Outputs:** A `.mid` file and piano roll visualizations of these files were produced from each approach. These files can be found and downloaded under `/kaggle/working/generated_midi/`.
* **Approach 1 (Pitch-Only) Result:** When listening to the generated MIDI, it was observed that although it had a melodic structure, it had a **completely monotonous rhythm** due to the fixed duration and timing. This is within expectations because the model did not learn rhythm.
* **Approach 2 (Event-Based) Result:** When listening to and examining the piano roll of the MIDI generated with this approach, it was observed that despite the rhythm still not being perfect, it had a **more variable and potentially more interesting rhythmic structure** compared to the first approach. The inclusion of time intervals broke the monotony.
* **Comparison:** The second approach showed potential to produce more acceptable results in terms of rhythm by attempting to model timing information. However, the complexity and originality of the melodies produced by both models are limited due to the simple LSTM architecture used and the limited training data (43 Aerosmith songs).

*(See the relevant Kaggle Notebook for detailed analysis of piano roll visualizations and generated MIDI files.)*

## Discussion and Implications

This project has successfully implemented two different approaches for music generation in MIDI format using LSTM networks:

* Modeling only the note sequence can produce melodic ideas but is rhythmically inadequate.
* Including time information (time shift) as discrete events in the modeling has the potential to increase rhythmic diversity, but assumptions still need to be made for note durations during conversion back to MIDI.
* In both approaches, the quality of the music produced depends on the capacity of the model, the amount/quality of the training data, and the musical representation used.
* This experiment has practically demonstrated the fundamental challenges of symbolic music generation (rhythm, duration, structure modeling).

## Environment and Libraries

* Platform: Kaggle Notebooks (with GPU)
* Main Libraries: `PyTorch`, `pretty_midi`, `numpy`, `matplotlib`, `pandas` (for data processing), `librosa` (for visualization).

## How to Run / Listen

When the relevant Kaggle Notebook is run, two `.mid` files (`lstm_generated_melody.mid` and `lstm_generated_melody_event.mid`) are created under the `/kaggle/working/generated_midi/` folder. These files can be downloaded from the Kaggle interface and listened to with any MIDI player or Digital Audio Workstation (DAW) software. The Notebook outputs also include piano roll visualizations of the generated melodies.) dataset found on Kaggle was used.
* **Subset:** Specifically, **43 MIDI files** from the folder containing MIDI files of the `Aerosmith` band were used for training. This was done to enable the model to learn a specific style (rock/pop) and to keep the dataset at a manageable size.

## Method

Two different LSTM-based models were developed and trained in the project:

### Approach 1: Pitch-Only Prediction

1. **Data Preprocessing:** Aerosmith MIDI files were read using the `pretty_midi` library. The note sequence (only pitch values) of the instrument with the most notes in each file was extracted, and notes from all files were combined to create a single long note sequence. A dictionary (vocabulary) was created from the unique notes in this sequence, and the note sequence was converted to integer indices. Input/output pairs (input: 100 notes, output: 101st note) were prepared for training using a sliding window method with fixed length (`sequence_length=100`).
2. **Model Architecture:** A PyTorch model (`MusicLSTM`) consisting of an Embedding layer, a 2-layer LSTM (`hidden_size=512`) followed by a Dropout layer, and a Linear layer producing output in the size of the vocabulary was defined.
3. **Training:** The model was trained on the prepared note sequences to learn to predict the next note using `CrossEntropyLoss` and the `Adam` optimizer.
4. **Generation:** Using the trained model, a new pitch sequence was generated iteratively, starting from a randomly selected seed sequence from the training data. Randomness was controlled by the `temperature` parameter during generation.
5. **Conversion to MIDI:** The generated note sequence was converted to a `.mid` file using `pretty_midi` by **assigning a fixed duration and fixed time interval to each note**. It is known that this approach does not represent rhythmic information.

### Approach 2: Event-Based (Pitch + Time Shift) Prediction

1. **Data Preprocessing:** When reading MIDI files, both note onsets (pitch) and **time differences (time shifts)** between consecutive notes were extracted. Time differences were quantized by dividing them into a certain number (e.g., 32) of discrete categories (quantile bins) according to their distribution. The music sequence was converted into a sequence of events in the form of `NOTE_pitch` and `TIME_binIndex`. A new, larger dictionary including both note and time events was created, and the event sequence was converted to integer indices. Event sequences of fixed length (`sequence_length_event=150`) and output pairs predicting the next event were prepared for training.
2. **Model Architecture:** The **same LSTM architecture** as in Approach 1 was used, but the dimensions of the Embedding and Linear layers were updated according to the size of the new and larger event dictionary (`n_vocab_event`).
3. **Training:** The model was trained on event sequences to learn to predict the next event (whether a note or a time interval) using the same loss function and optimizer.
4. **Generation:** Using the trained model, a new **event** sequence was generated iteratively, starting from a random seed event sequence.
5. **Conversion to MIDI:** The generated event sequence was interpreted:
   * When a `TIME_` event was predicted: The corresponding (average) duration was calculated and added to the current time in the MIDI. This duration was also stored as the potential duration of the next note.
   * When a `NOTE_` event was predicted: The note was placed at the current time, and its duration was assigned as the duration of the `TIME_` event that immediately preceded it (a simple rhythmic assumption).
   * Results were converted to a `.mid` file using `pretty_midi`.

## Results and Comparison

* **Outputs:** A `.mid` file and piano roll visualizations of these files were produced from each approach. These files can be found and downloaded under `/kaggle/working/generated_midi/`.
* **[Approach 1 (Pitch-Only) Result:](lstm_generated_melody.mid)** When listening to the generated MIDI, it was observed that although it had a melodic structure, it had a **completely monotonous rhythm** due to the fixed duration and timing. This is within expectations because the model did not learn rhythm. 
* **[Approach 2 (Event-Based) Result:](lstm_generated_melody_event.mid)** When listening to and examining the piano roll of the MIDI generated with this approach, it was observed that despite the rhythm still not being perfect, it had a **more variable and potentially more interesting rhythmic structure** compared to the first approach. The inclusion of time intervals broke the monotony.
* **Comparison:** The second approach showed potential to produce more acceptable results in terms of rhythm by attempting to model timing information. However, the complexity and originality of the melodies produced by both models are limited due to the simple LSTM architecture used and the limited training data (43 Aerosmith songs).

*(See the relevant Kaggle Notebook for detailed analysis of piano roll visualizations and generated MIDI files.)*

## Discussion and Implications

This project has successfully implemented two different approaches for music generation in MIDI format using LSTM networks:

* Modeling only the note sequence can produce melodic ideas but is rhythmically inadequate.
* Including time information (time shift) as discrete events in the modeling has the potential to increase rhythmic diversity, but assumptions still need to be made for note durations during conversion back to MIDI.
* In both approaches, the quality of the music produced depends on the capacity of the model, the amount/quality of the training data, and the musical representation used.
* This experiment has practically demonstrated the fundamental challenges of symbolic music generation (rhythm, duration, structure modeling).

## Environment and Libraries

* Platform: Kaggle Notebooks (with GPU)
* Main Libraries: `PyTorch`, `pretty_midi`, `numpy`, `matplotlib`, `pandas` (for data processing), `librosa` (for visualization).

## How to Run / Listen

When the relevant Kaggle Notebook is run, two `.mid` files ( [`lstm_generated_melody.mid`](lstm_generated_melody.mid) and [`lstm_generated_melody_event.mid`](lstm_generated_melody_event.mid) ) are created under the `/kaggle/working/generated_midi/` folder. These files can be downloaded from the Kaggle interface and listened to with any MIDI player or Digital Audio Workstation (DAW) software. The Notebook outputs also include piano roll visualizations of the generated melodies.
