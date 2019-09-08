import numpy
import tensorflow as tf

class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """ 
        ***CONSTRUCTOR***

        Args:
        
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).

        """
        
        self.hidden_forward = 42
        self.hidden_backward = 45
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        
        self.x = tf.placeholder(tf.int64, [None, self.max_length], 'X')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')
        self.target = tf.placeholder(tf.int32, [None, self.max_length], 'Tags')
        
        self.session = tf.Session()

    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.

        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        x = tf.sequence_mask(length_vector, maxlen=self.max_length)
        return tf.dtypes.cast(x, dtype=tf.float32)


    def build_inference(self):
        """
         Built the expression from (self.x, self.lengths) to (self.logits).

         I am using bi directional LSTM (Backward and Forward)

        """

        embedding = tf.get_variable('embeddings',[self.num_terms, 45])
        
        
        x_embedding = tf.nn.embedding_lookup(embedding, self.x)
        
        backward_cell = tf.contrib.rnn.LSTMCell(self.hidden_backward,use_peepholes=True,forget_bias=0.0)
        
        forward_cell = tf.contrib.rnn.LSTMCell(self.hidden_forward,use_peepholes=True,forget_bias=0.0)
        
        
        
        (forward_output, backward_output), temp = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, x_embedding, sequence_length=self.lengths, dtype=tf.float32)
        
        com_output = tf.concat([forward_output, backward_output], axis=-1)

        
        W = tf.get_variable("Weight", dtype=tf.float32, shape=[self.hidden_forward + self.hidden_backward, self.num_tags])

        b = tf.get_variable("bias", shape=[self.num_tags], dtype=tf.float32, initializer=tf.zeros_initializer())
        
        n_steps = tf.shape(com_output)[1]
        
        com_output = tf.reshape(com_output, [-1, self.hidden_forward + self.hidden_backward])
        
        predict = tf.matmul(com_output, W) + b
        
        self.logits = tf.reshape(predict, [-1, n_steps, self.num_tags])


    def run_inference(self, terms, lengths):
        """Evaluated self.logits given self.x and self.lengths.

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. 
        """

        logits = self.session.run(self.logits, {self.x: terms, self.lengths: lengths})
        return numpy.argmax(logits, axis=2)

    def build_training(self):
        """
        Preparing the class for training.
        
        """

        step_grad = tf.Variable(0,trainable=False)
        val = 1e-2
        learn_rate =  tf.train.exponential_decay(val, step_grad, 990, 0.9, staircase = False)
        loss_val = tf.contrib.seq2seq.sequence_loss(self.logits, self.target, self.lengths_vector_to_binary_matrix(self.lengths))
        loss = tf.reduce_mean(loss_val)
        self.train_op = tf.train.AdamOptimizer(learning_rate = learn_rate,beta1=0.9,beta2=0.999,epsilon=1e-11,use_locking=True,name='Adam').minimize(loss)
        self.session.run(tf.global_variables_initializer())




    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=1e-7):

        """Performed updates on the model given training data.

        This will be called with numpy arrays similar to the ones created in
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.

        Return:
                boolean. Returned True as I want the training to continue. 
                If you returned False (or do not return anyhting) then training will stop after
                the first iteration.
        """

        len = terms.shape[0]

        indices = numpy.random.permutation(len)

        for start in range(0, len, batch_size):
            last = min(len, start + batch_size)

            bat_x = terms[indices[start:last]] + 0
            bat_y = tags[indices[start:last]] + 0
            bat_z = lengths[indices[start:last]] + 0

            self.session.run(self.train_op, {self.x: bat_x, self.target: bat_y, self.lengths:bat_z})


        return True

    def evaluate (self,test_terms, test_tags, test_lengths):
        
        len = test_terms.shape[0]

        indices = numpy.random.permutation(len)

        for start in range(0, len, batch_size):
            last = min(len, start + batch_size)

            bat_x = test_terms[indices[start:last]] + 0
            bat_y = test_tags[indices[start:last]] + 0
            bat_z = test_lengths[indices[start:last]] + 0
            predict = run_inference(self, bat_x, bat_z)
      
            
        
        

  