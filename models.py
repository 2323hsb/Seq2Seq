import tensorflow as tf

class Seq2Seq(tf.keras.Model):
    def __init__(self, token_x_len, token_y_len, state_dim):
        super(Seq2Seq, self).__init__()
        self.token_x_len = token_x_len
        self.token_y_len = token_y_len
        self.state_dim = state_dim

        enc_in = tf.keras.layers.Input(shape=(None, self.token_x_len))
        enc_lstm_out, enc_state_h, enc_state_c  = tf.keras.layers.LSTM(units=self.state_dim, return_state=True)(enc_in)
        context_state = [enc_state_h, enc_state_c]
        self.encoder = tf.keras.Model(enc_in, [enc_lstm_out, context_state])

        dec_in = tf.keras.layers.Input(shape=(None, self.token_y_len))
        dec_in_h = tf.keras.layers.Input(shape=(self.state_dim,))
        dec_in_c = tf.keras.layers.Input(shape=(self.state_dim,))
        dec_states_in = [dec_in_h, dec_in_c]
        dec_lstm_out, dec_state_h, dec_state_c = tf.keras.layers.LSTM(units=self.state_dim, return_state=True, return_sequences=True)(dec_in, initial_state=dec_states_in)
        dec_out = tf.keras.layers.Dense(self.token_y_len, activation='softmax')(dec_lstm_out)
        self.decoder = tf.keras.Model([dec_in, dec_in_h, dec_in_c], [dec_out, dec_state_h, dec_state_c])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y, h, c):
        return self.decoder([y, h, c])

# class Attention(tf.keras.Model):
#     def __init__(self, token_x_len, token_y_len, state_dim):
#         super(Attention, self).__init__()
#         self.token_x_len = token_x_len
#         self.token_y_len = token_y_len
#         self.state_dim = state_dim

#         enc_in = tf.keras.layers.Input(shape=(None, self.token_x_len))
#         enc_lstm_out, enc_state_h, enc_state_c  = tf.keras.layers.LSTM(units=self.state_dim, return_state=True, return_sequences=True)(enc_in)
#         self.encoder = tf.keras.Model(enc_in, [enc_lstm_out, enc_state_h, enc_state_c])

#         dec_in = tf.keras.layers.Input(shape=(None, self.token_y_len))
#         dec_in_h = tf.keras.layers.Input(shape=(self.state_dim,))
#         dec_in_c = tf.keras.layers.Input(shape=(self.state_dim,))
#         dec_in_enc = tf.keras.layers.Input(shape=(self.token_x_len, self.state_dim))
#         dec_states_in = [dec_in_h, dec_in_c]
#         dec_lstm_out, dec_state_h, dec_state_c = tf.keras.layers.LSTM(units=self.state_dim, return_state=True, return_sequences=True)(dec_in, initial_state=dec_states_in)
#         attention = self.dot_product_attention(dec_in_enc, dec_lstm_out)
#         # softmax_out = tf.keras.layers.Dense(self.token_y_len, activation='softmax')(dec_lstm_out)
#         self.decoder = tf.keras.Model([dec_in, dec_in_h, dec_in_c, dec_in_enc], [dec_lstm_out, dec_state_h, dec_state_c])

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, y, h, c, e):
#         return self.decoder([y, h, c, e])

#     def dot_product_attention(self, enc_out, dec_out):
#         weights = tf.nn.softmax(tf.matmul(enc_out, dec_out, transpose_b=True))
#         return tf.matmul(enc_out, weights, transpose_a=True)