from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            radius_mean=float(request.form.get('radius_mean')),
            texture_mean = float(request.form.get('texture_mean')),
            smoothness_mean = float(request.form.get('smoothness_mean')),
            compactness_mean = float(request.form.get('compactness_mean')),
            concavity_mean = float(request.form.get('concavity_mean')),
            symmetry_mean = float(request.form.get('symmetry_mean')),
            fractal_dimension_mean = float(request.form.get('fractal_dimension_mean')),
            radius_se = float(request.form.get('radius_se')),
            texture_se = float(request.form.get('texture_se')),
            smoothness_se = float(request.form.get('smoothness_se')),
            compactness_se = float(request.form.get('compactness_se')),
            concavity_se = float(request.form.get('concavity_se')),
            concave_points_se = float(request.form.get('concave_points_se')),
            symmetry_se = float(request.form.get('symmetry_se')),
            fractal_dimension_se = float(request.form.get('fractal_dimension_se')),
            smoothness_worst = float(request.form.get('smoothness_worst')),
            compactness_worst = float(request.form.get('compactness_worst')),
            concavity_worst = float(request.form.get('concavity_worst')),
            symmetry_worst = float(request.form.get('symmetry_worst')),
            fractal_dimension_worst = float(request.form.get('fractal_dimension_worst')),
           
            
           
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=pred
        if result == 0:
            return render_template("result.html",final_result = "Breast Cancer is benign")
        elif result == 1:
            return render_template("result.html",final_result = "Breast Cancer is malignant")





if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True,port=5000)