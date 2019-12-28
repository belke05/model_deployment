import tensorflow as tf

def predict_text(start_string, model):
  print(tf.version.VERSION)
  vocab = ["\t","\n", " ", "!", '"', "#", "'", "(", ")", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", ":", ";", "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y" ,"z"]
  idx2char = vocab
  char2idx = {
    "\t": 0,"\n": 1, " ": 2, "!": 3, '"': 4, "#": 5, "'": 6, "(": 7, ")": 8, ",": 9,"-": 10,".": 11,"/": 12,"0": 13,"1": 14,"2": 15,"3": 16,"4": 17,"5": 18,"6": 19,"7": 20,"8": 21,":": 22,";": 23,
    "?": 24,'A': 25,'B': 26,'C': 27,'D': 28,'E': 29,'F': 30,'G': 31,'H': 32,'I': 33,'J': 34,'K': 35,'L': 36,'M': 37,'N': 38,'O': 39,'P': 40,'Q': 41,'R': 42,'S': 43,'T': 44,'U': 45,'V': 46,'W': 47,'X': 48,'Y': 49,'Z': 50,'a': 51,'b': 52,'c': 53,'d': 54,'e': 55,'f': 56,'g': 57,'h': 58,'i': 59,'j': 60,'k': 61,'l': 62,'m': 63,'n': 64,'o': 65,'p': 66,'q': 67,'r': 68,'s': 69,'t': 70,'u': 71,'v': 72,'w': 73,'x': 74,'y': 75, 'z': 76
  }

  # Evaluation step (generating text using the learned model)
  
  # Number of characters to generat
  print(start_string)
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the word returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # We pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))