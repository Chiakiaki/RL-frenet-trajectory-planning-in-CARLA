#!/bin/bash --login
#$ -cwd
#$ -l v100=1         # A 4-GPU request (v100 is just a shorter name for nvidia_v100)
#$ -pe smp.pe 8     # Let's use the 8 CPUs per GPU (32 cores in total)

# CSF3 configuration part ===========
module load compilers/gcc/8.2.0
# module load apps/binapps/anaconda3/2019.07 #bug, the csf3 seems run this in computation node, causing inconsistency
module load libs/cuda/10.1.243
module load libs/gcc/glx/1.5 libs/gcc/glew/2.1.0 libs/gcc/glm/0.9.9.4 libs/gcc/glog/0.4.0
module load mpi/gcc/openmpi/4.1.0
conda activate ~/tensorflow115
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export CARLA_ROOT=~/CARLA_0.9.9.2/
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/libjpeg/jpeg-8d1/lib
export OMP_NUM_THREADS=8
# ===================================

# portRange="80-81"
# rangeStart=$(echo ${portRange} | awk -F '-' '{print $1}')
# rangeEnd=$(echo ${portRange} | awk -F '-' '{print $2}')


# see whether the port is free
function Listening {
   TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
   UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
   (( Listeningnum = TCPListeningnum + UDPListeningnum ))
   if [ $Listeningnum == 0 ]; then
       echo "0"
   else
       echo "1"
   fi
}

# random range
function random_range {
   shuf -i $1-$2 -n1
}

# get port, two input: $1 $2
PORT=-1
function get_random_port {
   if [ $1 -gt $2 ]; then
      echo "error: please check port range"
      exit
   fi
   temp1=$1
   while [ $PORT == -1 ]  && [ $temp1 -le $2 ]; do
       if [ `Listening $temp1` == 0 ]; then
              PORT=$temp1
       else
              ((temp1 = temp1 + 1))
       fi
   done
   if [ $PORT == -1 ]; then
      echo "No avaliable port"
      exit
   fi
   echo "$PORT"
}
# main
port=$(get_random_port 2000 4000);
port_tm=$(get_random_port 8000 9000);
# Above part is blog code, for getting a port.
# Now Carla part:
echo "using port $port $port_tm"
(DISPLAY= ~/CARLA_0.9.9.2/CarlaUE4.sh -world-port=$port -opengl &)

#try change map to town04
tmpmsg="123"
while [ -n "$tmpmsg" ]; do #see whether it is null
   sleep 1
   tmpmsg=`python3 ~/CARLA_0.9.9.2/PythonAPI/util/config.py --map Town04 -p $port | grep time-out`
   echo $tmpmsg
done

# ===================================
cd ~/RL-frenet-trajectory-planning-in-CARLA/
python3 ./run_BDPL.py -p $port --tm_port $port_tm --cfg_file=tools/cfgs/config_ddpg.yaml --agent_id=44440 --env=CarlaGymEnv-v5 --learning_rate=7e-4 --planner_mode=continuous_catagorical



#now close the server pid
tmp=$(ps | grep CarlaUE4-Linux)
tmp=($tmp)
kill $tmp
sleep 1
kill $tmp
