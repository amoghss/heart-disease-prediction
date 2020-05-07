from flask import Flask , render_template , request , jsonify
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import Normalizer,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('app.html')

@app.route('/',methods=['POST'])
def predict():
   features=[]

   features = [int(x) for x in request.form.values()]
   p = pd.DataFrame([np.array(features)])

   #cp = p[2][0]
   a=joblib.load('pickle/cp.pkl')
   cp= a.transform(p[2].values.reshape(-1, 1)).toarray()

   #restecg=p[6][0]
   a=joblib.load('pickle/restecg.pkl')
   restecg= a.transform(p[6].values.reshape(-1, 1)).toarray()

   #slope=p[10][0]
   a=joblib.load('pickle/slope.pkl')
   slope= a.transform(p[10].values.reshape(-1, 1)).toarray()

   #ca=p[11][0]
   a=joblib.load('pickle/ca.pkl')
   ca= a.transform(p[11].values.reshape(-1, 1)).toarray()

   #thal=p[12][0]
   a=joblib.load('pickle/thal.pkl')
   thal= a.transform(p[12].values.reshape(-1, 1)).toarray()

   cp=pd.DataFrame(cp,columns=['cp_1','cp_2','cp_3','cp_4'])
   cp.drop('cp_4',axis=1,inplace=True)
   cp=cp.astype(int)

   restecg=pd.DataFrame(restecg,columns='resecg_1 resecg_2 resecg_3'.split())
   restecg.drop('resecg_3',axis=1,inplace=True)
   restecg=restecg.astype(int)

   slope=pd.DataFrame(slope,columns='slope_1 slope_2 slope_3'.split())
   slope.drop('slope_3',axis=1,inplace=True)
   slope=slope.astype(int)

   ca=pd.DataFrame(ca,columns='ca_1 ca_2 ca_3 ca_4 ca_5'.split())
   ca.drop('ca_5',axis=1,inplace=True)
   ca=ca.astype(int)

   thal=pd.DataFrame(thal,columns='thal_1 thal_2 thal_3 thal_4'.split())
   thal.drop('thal_4',axis=1,inplace=True)
   thal=thal.astype(int)

   p = p.drop([2 , 6 ,  10 ,  11 ,  12] ,  axis=1)

   p = pd.concat([p,cp,restecg,slope,ca,thal],axis=1)

   norm = joblib.load('pickle/normalize.pkl')
   p=norm.transform(p)

   model=joblib.load('pickle/model.pkl')

   answer=model.predict(p)[0]

   print(answer)

   if answer == 1:
      return render_template('app.html', prediction='You Might be Suffering from a Heart Disease')

   return render_template('app.html', prediction='You Seem All Fine Congratulations')

if __name__ == '__main__':
   app.run(debug = True)