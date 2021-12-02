source /data/iot/notebook/envs/ai/bin/activate
cd /data/iot/notebook/src

#nohup python -u inference.py gbm > ./logs/gbm.log &
#nohup python -u inference.py rf > ./logs/rf.log &
#nohup python -u inference.py bayes > ./logs/bayes.log &
#nohup python -u inference.py deeplearn > ./logs/deeplearn.log &

#done
#nohup python -u inference.py glm > ./logs/glm.log &

#on-going
#nohup python -u inference.py xgboost > ./logs/xgboost.log &
nohup python -u inference.py svm > ./logs/svm.log &
