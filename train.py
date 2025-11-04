from dataset import test_info,prepare_data, visualize
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import joblib

def linear_regression(X_train,y_train,val_set,rt1_val):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score

    model = LinearRegression()
    model.fit(X_train, y_train)

    #predikcija za val set
    y_pred = model.predict(val_set)

    #evaluacija
    mse = mean_squared_error(rt1_val, y_pred)
    r2 = r2_score(rt1_val, y_pred)
    mae = mean_absolute_error(rt1_val,y_pred)
    evs = explained_variance_score(rt1_val, y_pred)
    
    print("\nLinear regression")
    print("-----------------------------")
    print("MSE:", mse)
    print("MAE:",mae)
    print("R²:", r2)
    print("Explained Variance:", evs)
    print("-----------------------------")

    plt.figure(figsize=(8,6))
    plt.scatter(range(len(rt1_val)), rt1_val, color='blue', s=3, label='Stvarne vrijednosti')
    plt.scatter(range(len(y_pred)), y_pred, color='red', s=3, label='Predikcije')
    plt.title('Linear regression')
    plt.text(0.95, 0.01, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.legend()
    plt.show()

    return {"model": model,"params": model.get_params()}


def ridge_regression(X_train, y_train, val_set, rt1_val):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,explained_variance_score
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(val_set)

    mse = mean_squared_error(rt1_val, y_pred)
    mae = mean_absolute_error(rt1_val, y_pred)
    r2 = r2_score(rt1_val, y_pred)
    evs = explained_variance_score(rt1_val, y_pred)

    print("\nRidge regression")
    print("-----------------------------")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R²:", r2)
    print("Explained Variance:", evs)
    print("-----------------------------")

    plt.figure(figsize=(8,6))
    plt.scatter(range(len(rt1_val)), rt1_val, color='blue', s=3, label='Stvarne vrijednosti')
    plt.scatter(range(len(y_pred)), y_pred, color='red', s=3, label='Predikcije')
    plt.title('Ridge regression')
    plt.legend()
    plt.show()

    return {"model": model,"params": model.get_params()}

def xgboost_(X_train,y_train,val_set,rt1_val,n_est,lr,depth):
    from xgboost import XGBRegressor
    import sklearn
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score

    model = XGBRegressor(
    objective='reg:squarederror',
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(val_set)

    mse = mean_squared_error(rt1_val, y_pred)
    r2 = r2_score(rt1_val, y_pred)
    mae = mean_absolute_error(rt1_val,y_pred)
    evs = explained_variance_score(rt1_val, y_pred)

    print("\nXGBoost")
    print("-----------------------------")
    print("MSE:", mse)
    print("MAE:",mae)
    print("R²:", r2)
    print("Explained Variance:", evs)
    print(f"depth={depth}, lr={lr}, n_est={n_est}")
    print("-----------------------------")
    
    plt.figure(figsize=(8,6))
    plt.scatter(range(len(rt1_val)), rt1_val, color='blue', s=3, label='Stvarne vrijednosti')
    plt.scatter(range(len(y_pred)), y_pred, color='red', s=3, label='Predikcije')
    plt.title('XGBoost')
    plt.text(0.95, 0.01, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.legend()
    plt.show()

    return {"model": model,"params": model.get_params()}

def NN_reg(X_train,y_train,val_set,rt1_val,h_layers,act,lr):
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score

    model = MLPRegressor(hidden_layer_sizes=h_layers,activation=act,solver='adam',max_iter=1000000,learning_rate_init=lr)
    model.fit(X_train,y_train)
    y_pred = model.predict(val_set)
    
    mse = mean_squared_error(rt1_val, y_pred)
    r2 = r2_score(rt1_val, y_pred)
    mae = mean_absolute_error(rt1_val,y_pred)
    evs = explained_variance_score(rt1_val, y_pred)
    
    print("\nNeural network regression")
    print("-----------------------------")
    print("MSE:", mse)
    print("MAE:",mae)   
    print("R²:", r2)
    print("Explained Variance:", evs)
    print(f"hidden_layers={h_layers}, lr={lr}, activation={act}")
    print("-----------------------------")
    
    plt.figure(figsize=(8,6))
    plt.scatter(range(len(rt1_val)), rt1_val, color='blue', s=3, label='Stvarne vrijednosti')
    plt.scatter(range(len(y_pred)), y_pred, color='red', s=3, label='Predikcije')
    plt.title('Neural network')
    plt.text(0.95, 0.01, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.legend()
    plt.show()
    
    values = [mse,mae,r2]
    return values,{"model": model,"params": model.get_params()}

def SVR(X_train,y_train,val_set,rt1_val):
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score

    model = SVR(kernel='rbf', C=1.0, epsilon=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(val_set)
    
    mse = mean_squared_error(rt1_val, y_pred)
    r2 = r2_score(rt1_val, y_pred)
    mae = mean_absolute_error(rt1_val,y_pred)
    evs = explained_variance_score(rt1_val, y_pred)
    
    print("\nSVR")
    print("-----------------------------")
    print("MSE:", mse)
    print("MAE:",mae)   
    print("R²:", r2)
    print("Explained Variance:", evs)
    print("-----------------------------")

    plt.figure(figsize=(8,6))
    plt.scatter(range(len(rt1_val)), rt1_val, color='blue', s=3, label='Stvarne vrijednosti')
    plt.scatter(range(len(y_pred)), y_pred, color='red', s=3, label='Predikcije')
    plt.title('SVR')
    plt.legend()
    plt.text(0.95, 0.01, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.show()

    return {"model": model,"params": model.get_params()}

def grid_search(model,X_train,y_train):
    param_grid = {
    'hidden_layer_sizes': [    
        (64, 32), (128, 64),      
        (128, 64, 32)          
    ],
    'activation': [
        'identity', 'tanh', 'logistic'
    ],
    'learning_rate_init': [
        0.1,0.01, 0.05 
    ],
    'solver': [
        'adam'  
    ],
    'max_iter': [1000]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Najbolji parametri:", grid_search.best_params_)
    print("Najbolji rezultat (neg MSE):", grid_search.best_score_)

def test(model_, X_test,y_test):
    from sklearn.neural_network import MLPRegressor
    import sklearn
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score

    model = model_["model"]
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print("\nNeural network regression")
    print("-----------------------------")
    print("MSE:", mse)
    print("MAE:",mae)   
    print("R²:", r2)
    print("Explained Variance:", evs)
    print("-----------------------------")

    plt.figure(figsize=(8,6))
    plt.scatter(range(len(y_test)), y_test, color='blue', s=3, label='Stvarne vrijednosti')
    plt.scatter(range(len(y_pred)), y_pred, color='red', s=3, label='Predikcije')
    plt.title('\nNeural network regression')
    plt.legend()
    plt.text(0.95, 0.01, f'MSE: {mse:.3f}\nMAE: {mae:.3f}',
         horizontalalignment='right',
         verticalalignment='bottom',
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
    plt.show()

    return model

def train(return_t1,train_set,val_set):
    n = len(return_t1)
    prvi = int(0.7 * n)
    drugi = int(0.85 * n)

    rt1_train = return_t1.iloc[:prvi]  #podijela ciljnih varijabli
    rt1_val = return_t1.iloc[prvi:drugi]
    rt1_test = return_t1.iloc[drugi:]

    #lreg_params = linear_regression(train_set,rt1_train,val_set,rt1_val)
    #ridge_params = ridge_regression(train_set,rt1_train,val_set,rt1_val)
    #xgb_params = xgboost_(train_set,rt1_train,val_set,rt1_val,7,0.05,10)
    #mlp_reg = NN_reg(train_set,rt1_train,val_set,rt1_val,h_layers=(128,64,32),act='tanh',lr=0.01)
    #svr = SVR(train_set,rt1_train,val_set,rt1_val)

    #values = [float('inf'),float('inf')]
    #najbolji = None
    '''
    for act in ['tanh', 'logistic']:  # ispravljeno
        for lr in [0.05, 0.1, 0.5, 1]:
            for h_layers in [(128,128),(128,64,32), (128,128,128,64,32), (256,256,128,128,64,32)]:
                
                val, model = NN_reg(train_set, rt1_train, val_set, rt1_val, h_layers, act, lr)
                
                if val[0] < values[0] and val[1] < values[1]:
                    values[0] = val[0]
                    values[1] = val[1]
                    najbolji = {'activation': act, 'learning_rate': lr, 'hidden_layers': h_layers}
    '''               

    #print("Najbolji parametri:", najbolji)
    #print("Najmanji MSE:", values[0])
    #model = MLPRegressor()
    #grid_search(model,train_set,rt1_train)

    _, model_ = NN_reg(train_set,rt1_train,val_set,rt1_val,h_layers=(128,64,32),act='tanh',lr=0.01)

    return model_, rt1_test

if __name__=="__main__":
    return_t1,train_set,val_set,test_set = prepare_data("Data/btc_historical_price.parquet")
    model, rt1_test = train(return_t1,train_set,val_set)
    model_params = test(model,test_set,rt1_test)
    #joblib.dump(model_params, "best_model.pkl")