from flask import Flask
from flask import render_template
from flask import request
import sarimax
import analysis
import linearAug
import polynomialAug
import arimaa
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():
    disp=True
    if request.method == 'POST':
        disp=False
        st=request.form['state']
        med=request.form['medicine']
        res=sarimax.sarimax_model(st,med)
        linearAug.linear(st,med)
        polynomialAug.poly(st,med)
        arimaa.arima(st,med)
    return render_template('form.html',disp=disp)


@app.route('/analysis',methods=['GET','POST'])
def analyse():
    disp=True
    if request.method == 'POST':
        disp=False
        st=request.form['state']
        res=analysis.analyse(st)
    return render_template('analyse_form.html',disp=disp)



app.run(debug=True)

