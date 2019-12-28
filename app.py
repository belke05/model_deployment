import flask, requests, pickle, joblib
from flask_cors import CORS, cross_origin
import numpy as np
import json

app = flask.Flask(__name__)

# Load the model
model = joblib.load("./linear_regression_model.pkl")
classifier = joblib.load("./classifier.pkl")


cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
def my_index():
  return "<h1>Welcome to our server !!</h1>"


@app.route("/predict_salary", methods=['GET', 'POST'])
def makePrediction():
  if flask.request.method == "GET":
    years_experience = flask.request.args.get('experience')
    years_experience = np.array(np.float(years_experience)).reshape(-1,1)
    print(years_experience, 'years of experience')
    prediction = model.predict(years_experience)
    prediction = prediction.flatten()[0]
    print(prediction, 'predictions')
    return flask.jsonify({'prediction': prediction})
  if flask.request.method == "POST":
    print(flask.request)
    form = flask.request.get_json(force=True)
    print("here")
    formkeys = [key for key in form.keys()]
    formvalues = [np.float(value) for value in form.values()]
    years_experience = formvalues[0]
    years_experience = np.array(np.float(years_experience)).reshape(-1, 1)
    prediction = model.predict(years_experience)
    prediction = prediction.flatten()[0]
    return flask.jsonify({'prediction': prediction})
  

@app.route("/predict_admission", methods=['POST'])
def makeAdmissionPrediction():
  # ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
  form = flask.request.json
  formkeys = [key for key in form.keys()]
  formvalues = [np.float(value) for value in form.values()]
  rndm_vals = np.array(formvalues).reshape(1, -1)
  print(rndm_vals)
  print(form)
  # 337 118 4 4.5 4.5 9.65 1 
  # GRE Score	TOEFL Score	University Rating	SOP	LOR	CGPA	Research
  # change for non admission is 1 change of admission is 0
  prediction = classifier.predict(np.array(rndm_vals))
  prediction = str(prediction.flatten()[0])
  print(prediction, 'predictions')
  return flask.jsonify({'prediction': prediction})


@app.route("/model.json", methods=['GET'])
def returnModel():
  # read file
  with open('model.json', 'r') as myfile:
      data=myfile.read()
  # parse file
  obj = json.loads(data)
  return flask.jsonify(obj)

@app.route("/group1-shard1of4.bin")
def returnShard1():
  with open('group1-shard1of4.bin', 'rb') as myshard:
    data=myshard.read()
    response = flask.make_response(data)
    response.headers.set('Content-Type', 'application/octet-stream')
    # response.headers.set('Content-Disposition', 'attachment', filename='np-array.bin')
    return response

@app.route("/group1-shard2of4.bin")
def returnShard2():
  with open('group1-shard2of4.bin', 'rb') as myshard:
    data=myshard.read()
    response = flask.make_response(data)
    response.headers.set('Content-Type', 'application/octet-stream')
    # response.headers.set('Content-Disposition', 'attachment', filename='np-array.bin')
    return flask.send_file(myshard)

@app.route("/group1-shard3of4.bin")
def returnShard3():
  with open('group1-shard3of4.bin', 'rb') as myshard:
    data=myshard.read()
    response = flask.make_response(data)
    response.headers.set('Content-Type', 'application/octet-stream')
    # response.headers.set('Content-Disposition', 'attachment', filename='np-array.bin')
    return response

@app.route("/group1-shard4of4.bin")
def returnShard4():
  with open('group1-shard4of4.bin', 'rb') as myshard:
    data=myshard.read()
    response = flask.make_response(data)
    response.headers.set('Content-Type', 'application/octet-stream')
    # response.headers.set('Content-Disposition', 'attachment', filename='np-array.bin')
    return response
  

if __name__ == '__main__':
  app.run(debug=True, port=7000)