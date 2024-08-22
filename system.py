import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge

class ML_System_Regression():
    def __init__(self):
        pass

    def load_data(self):
        path = "C:/Users/Alber/OneDrive/Documentos/USTA 20242/INTELIGENCIA ARTIFICIAL/CORTE 1/SEMANA 3/"
        dataset = pd.read_csv(path + "dataset_APP.csv",header = 0,sep=";",decimal=",") 
        prueba = pd.read_csv(path + "prueba_APP.csv",header = 0,sep=";",decimal=",")
        return dataset, prueba
    
    def processing_data(self,dataset,prueba):
        ft = pd.DataFrame(dataset.isnull().sum()).reset_index()
        ft.columns = ["Variable","Faltantes"]
        ft["% Faltantes"] = ft["Faltantes"] * 100 / dataset.shape[0]
        formato = pd.DataFrame({'Variable': list(dataset.columns), 'Formato': dataset.dtypes })
        ft = pd.merge(ft,formato,on=["Variable"],how="left")
        
        # Set de entrenamiento
        cuantitativas = list(formato.loc[formato["Formato"]!="object","Variable"])
        cuantitativas = [x for x in cuantitativas if x not in ["Email","Address"]]
        cuantitativas = [x for x in cuantitativas if x not in ["Email","Address","price"]]
        categoricas = list(formato.loc[formato["Formato"]=="O","Variable"])
        categoricas = [x for x in categoricas if x not in ["Email","Address"]]
        numericas = dataset.get(cuantitativas)
        dum = pd.DataFrame({})
        for p in categoricas:
            dm = pd.get_dummies(dataset[p])
            for k in dm.columns:
                dm[k] = dm.apply(lambda row: 1 if row[k]==True else 0, axis = 1)
            dum = pd.concat([dum,dm],axis = 1)
        base_modelo = pd.concat([numericas,dum],axis = 1)
        base_modelo["y"] = dataset["price"].copy()
        
        # Set de prueba
        numericas2 = prueba.get(cuantitativas)
        dum2 = pd.DataFrame({})
        for p in categoricas:
            dm = pd.get_dummies(prueba[p])
            for k in dm.columns:
                dm[k] = dm.apply(lambda row: 1 if row[k]==True else 0, axis = 1)
            dum2 = pd.concat([dum2,dm],axis = 1)
        base_predicciones = pd.concat([numericas2,dum2],axis = 1)
        base_predicciones["y"] = prueba["price"].copy()
        
        # X, y
        covariables = [x for x in base_modelo.columns if x not in ["y"]]
        X = base_modelo.get(covariables).copy()
        y = base_modelo.get(["y"])
        X_nuevo = base_predicciones.get(covariables).copy()
        y_nuevo = base_predicciones.get(["y"])
        
        # Retornos
        return X, y, X_nuevo, y_nuevo
    
    def forecast(self,grilla_completa,X_nuevo):
        yhat_nuevo = grilla_completa.predict(X_nuevo)
        return yhat_nuevo
    
    def training_model(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
        modelo1 = Ridge()
        parametros = {'alpha':np.arange(0.1,5.1,0.1)}
        grilla = GridSearchCV(estimator=modelo1,param_grid=parametros,cv=5,scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=-1)
        grilla.fit(X_train,y_train)
        
        modelo2 = Ridge()
        grilla2 = GridSearchCV(estimator=modelo2,param_grid=parametros,cv=5,scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=-1)
        grilla2.fit(X_test,y_test)
        
        yhat_test = grilla.predict(X_test)
        yhat_train = grilla2.predict(X_train)
        e1 = np.sqrt(mean_squared_error(y_test,yhat_test))
        e2 = np.sqrt(mean_squared_error(y_train,yhat_train))
        
        if (np.abs(e1 - e2)<5) & (np.abs(grilla.best_params_['alpha'] - grilla2.best_params_['alpha']) < 0.5):
            modelo_completo = Ridge()
            parametros = {'alpha':np.arange(0.1,5.1,0.1)}
            grilla_completa = GridSearchCV(estimator=modelo_completo,param_grid=parametros,cv=5,scoring=make_scorer(mean_squared_error,greater_is_better=False),n_jobs=-1)
            grilla_completa.fit(X,y)
        else:
            grilla_completa = Ridge()
            grilla_completa.fit(X,y)
        return grilla_completa

    def accuracy(self,ytrue,yhat):
        var = np.abs((ytrue-yhat)/ytrue)
        return 100 * np.mean( var<= 0.02 )
    
    def evaluate_model(self,y_nuevo,yhat_nuevo):
        return self.accuracy(y_nuevo,yhat_nuevo)
    
    def ML_Flow_regression(self):
        try:
            dataset, prueba = self.load_data()
            X, y, X_nuevo, y_nuevo = self.processing_data(dataset,prueba)
            grilla_completa = self.training_model(X,y)
            yhat_nuevo = self.forecast(grilla_completa,X_nuevo)
            metric = self.evaluate_model(y_nuevo,yhat_nuevo)
            return {'success':True, 'accuracy':metric }
        except Exception as e:
            return {'succes':False, 'message':str(e)}