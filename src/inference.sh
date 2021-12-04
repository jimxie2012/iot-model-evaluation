source /data/iot/notebook/envs/ai/bin/activate
cd /data/iot/notebook/src

python -u inference.py gbm
python -u inference.py rf
python -u inference.py bayes 
python -u inference.py deeplearn
python -u inference.py glm 
python -u inference.py xgboost 
python -u inference.py svm 
