setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python train.py --config configs/admind-rl.yaml

app:
	streamlit run app.py
