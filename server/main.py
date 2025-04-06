import uuid
import time
import pandas as pd

from io import StringIO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from chart import Charter
from store import Store
from trade_test import TradingTester
from value_iter import ValueIterationExp
from trade_env import TradingEnvironment


app = FastAPI()
store = Store()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World!"}

@app.get("/models")
async def get_models():
    return {
        "status": "success",
        "models": list(store.models.keys())
    }

@app.get("/results/{model_id}")
async def get_results(model_id: str):
    try:    
        model = store.get_model(model_id)
        training_data = model["env"].training_data
        data = Charter.chart_data(training_data)

        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

@app.post("/train")
async def train(
    files: list[UploadFile] = File(...), 
    expected_return_step_size: int = 5,
    round_num_digits: int = 3,
    bank: float = 100.0,
    transaction_cost: float = 0.005,
    action_interval: int = 25,
    action_total: int = 100,
    lamb: float = 0.5, 
    gamma: float = 0.99,
    epsilon: float = 1e-3
):
    try:
        start_time = time.time()
        
        training_data = []
        for file in files:
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
            training_data.append(df)

        env = TradingEnvironment(training_data, 
                                 expected_return_step_size, 
                                 round_num_digits, 
                                 transaction_cost,
                                 bank,
                                 action_interval,
                                 action_total)

        policy, V, steps, updates = ValueIterationExp.run(env,
                                                          lamb=lamb, 
                                                          gamma=gamma, 
                                                          epsilon=epsilon)
        time_elapsed = time.time() - start_time

        model_id = str(uuid.uuid4())
        model = {
            "id": model_id,
            "env": env,
            "policy": policy.tolist(),
            "value_function": V.tolist(),
            "steps": steps,
            "updates": updates,
        } 
        store.set_model(model_id, model)
        
        return {
            "status": "success",
            "time_elapsed": time_elapsed,
            "model_id": model_id,
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
    
@app.post("/test")
async def test(
    model_id: str,
    time_frame: int = 60,
    n_episodes: int = 50000
):
    try:
        model = store.get_model(model_id)
        env = model["env"]
        policy = model["policy"]
        result, history_exec = TradingTester.test(env, policy, time_frame, n_episodes)
        # Generate the graphs with the results
        results = {
            "result": result,
            "history_exec": history_exec
        }
        store.set_results(model_id, results)

        return {
            "status": "success",
            "model_id": model["id"],
        }
    except KeyError as e:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )


@app.post("/run")
async def run(
    files: list[UploadFile] = File(...), 
    expected_return_step_size: int = 5,
    round_num_digits: int = 3,
    bank: float = 100.0,
    transaction_cost: float = 0.005,
    action_interval: int = 25,
    action_total: int = 100,
    lamb: float = 0.5, 
    gamma: float = 0.99,
    epsilon: float = 1e-3
):
    try:
        start_time = time.time()
        
        training_data = []
        for file in files:
            contents = await file.read()
            df = pd.read_csv(StringIO(contents.decode('utf-8')))
            training_data.append(df)

        env = TradingEnvironment(training_data, 
                                 expected_return_step_size, 
                                 round_num_digits, 
                                 transaction_cost,
                                 bank,
                                 action_interval,
                                 action_total)

        policy, V, steps, updates = ValueIterationExp.run(env,
                                                          lamb=lamb, 
                                                          gamma=gamma, 
                                                          epsilon=epsilon)
        time_elapsed = time.time() - start_time

        model_id = str(uuid.uuid4())
        model = {
            "id": model_id,
            "env": env,
            "policy": policy.tolist(),
            "value_function": V.tolist(),
            "steps": steps,
            "updates": updates,
        } 
        store.set_model(model_id, model)

        # test and then chart the results

        data = Charter.chart_data(training_data)

        return {
            "status": "success",
            "time_elapsed": time_elapsed,
            "model_id": model_id,
            "data": data
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )
