from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
file  = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/about/')
def about():
    return render_template('about us.html')
@app.route("/" , methods=['GET','POST'])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])
        
        input_features = [fever,bodyPain,age,runnyNose,diffBreath]
        inf_prob=clf.predict_proba([input_features])[0][1]
        print(inf_prob*100)
        return render_template('show.html',inf=round( inf_prob*100,2))   
    return render_template('index.html') 
    # return "Hello, World!" +str(inf_prob)

if __name__ == '__main__':
    app.run(debug= True)
    