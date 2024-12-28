conda activate carla

watch -n 1 -d nvidia-smi

./CarlaUE4.sh -quality-level=Low --no-rendering

python ~/project/PythonAPI/util/config.py --map Town12


