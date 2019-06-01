import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class TranslationModelTrainer:

    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.start_token = "_"
        self.end_token = "|"
        self.encoder = None
        self.attention_layer = None
        self.decoder = None

    def train(self, **kwargs):
        input_tensor, target_tensor, inp_lang, targ_lang = self._load_dataset(self.data_path)
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                        target_tensor,
                                                                                                        test_size=0.2)

        BUFFER_SIZE = len(input_tensor_train)
        BATCH_SIZE = 64
        steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
        embedding_dim = 256
        units = 1024
        vocab_inp_size = len(inp_lang.word_index) + 1
        vocab_tar_size = len(targ_lang.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        # To compile models
        example_input_batch, example_target_batch = next(iter(dataset))
        self.encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
        sample_hidden = encoder.initialize_hidden_state()
        encoder(example_input_batch, sample_hidden)

        self.attention_layer = BahdanauAttention(10)
        attention_layer(sample_hidden, sample_output)

        self.decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        decoder(tf.random.uniform((64, 1)), sample_hidden, sample_output)

        checkpoint_prefix = os.path.join(self.model_path, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=encoder,
                                         decoder=decoder)

        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        EPOCHS = 10

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(optimizer,
                                             loss_object,
                                             inp,
                                             targ,
                                             enc_hidden,
                                             targ_lang,
                                             BATCH_SIZE)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))



    def _loss_function(self, loss_object_ref, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object_ref(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, optimizer_ref, loss_object_ref, inp, targ, enc_hidden, targ_lang, BATCH_SIZE):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index[self.start_token]] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                loss += self._loss_function(loss_object_ref, targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer_ref.apply_gradients(zip(gradients, variables))
        return batch_loss


    def _tokenize(self, lang, language="en"):
        if language == "en":
            lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        if language == "ch":
            lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', char_level=True)

        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
        return tensor, lang_tokenizer

    def _load_dataset(self, path):
        # creating cleaned input, output pairs
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        targ_lang, inp_lang = data["en"], data["ch"]

        input_tensor, inp_lang_tokenizer = tokenize(inp_lang, language="ch")
        target_tensor, targ_lang_tokenizer = tokenize(targ_lang, language="en")

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def _max_length(self, tensor):
        return max(len(t) for t in tensor)



