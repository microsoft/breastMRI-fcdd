## Example of running task #1 for Balanced Detection for all models in parallel tmux sessions and multiple GPUs

# # # TASK #1 -- Balanced Detection

# # FCDD
# # Create a tmux session
sessname="paper_task0_fcdd_"
declare -a gpus=(1 2 4 5 6)

# Create a tmux session for each fold and run a command in each session in parallel
for i in 0 1 2 3 4
do
    tmux new-session -d -s $sessname$i
    tmux send-keys -t $sessname$i "bash" C-m
    tmux send-keys -t $sessname$i "conda activate fcdd_conda" C-m
    tmux send-keys -t $sessname$i "cd /home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/fcdd/runners" C-m
    tmux send-keys -t $sessname$i "python run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_$i --net FCDD_CNN224_VGG_NOPT --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective fcdd --logdir-suffix task0_no_aug_200_fcdd_$i --gpu ${gpus[$i]}" C-m
    echo "TMUX session $sessname$i running."
done

# # FCDD REF (Symmetric)
# Create a tmux session
sessname="t0_ref"
declare -a gpus=(2 3 4 5 7)
# # # Create a tmux session for each fold and run a command in each session in parallel
for i in 0 1 2 3 4
    do
        tmux new-session -d -s $sessname$i
        tmux send-keys -t $sessname$i "bash" C-m
        tmux send-keys -t $sessname$i "conda activate fcdd_env" C-m
        tmux send-keys -t $sessname$i "cd /home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/fcdd" C-m
        tmux send-keys -t $sessname$i "python runners/run_scans_refs.py --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_$i --net FCDD_REF_CNN224_VGG_NOPT --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective fcddrefs --logdir-suffix task0_ref_rand_$i --gpu ${gpus[$i]}" C-m
        echo "TMUX session $sessname$i running."
   done

# # BCE
# # Create a tmux session
sessname="paper_task0_bce_"

# Define an array for GPU numbers
declare -a gpus=(1 2 4 5 6)

# Create a tmux session for each fold and run a command in each session in parallel
for i in 0 1 2 3 4
do
    tmux new-session -d -s $sessname$i
    tmux send-keys -t $sessname$i "bash" C-m
    tmux send-keys -t $sessname$i "conda activate fcdd_conda" C-m
    tmux send-keys -t $sessname$i "cd /home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/fcdd/runners" C-m
    tmux send-keys -t $sessname$i "python run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_$i --net VGG_BCE_CROP --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective bce --logdir-suffix task0_no_aug_200_bce_$i --gpu ${gpus[$i]}" C-m
    echo "TMUX session $sessname$i running."
done


# # HSC
sessname="paper_task0_hsc_"

# Define an array for GPU numbers
declare -a gpus=(1 2 7 4 5)

# Create a tmux session for each fold and run a command in each session in parallel
for i in 0 1 2 3 4
do
    tmux new-session -d -s $sessname$i
    tmux send-keys -t $sessname$i "bash" C-m
    tmux send-keys -t $sessname$i "conda activate fcdd_conda" C-m
    tmux send-keys -t $sessname$i "cd /home/felipeoviedoperhavec/azurefiles/projects/FH.MRIbreast_fcdd/python/fcdd/runners" C-m
    tmux send-keys -t $sessname$i "python run_custom.py --supervise-mode other --datadir /home/felipeoviedoperhavec/ssdprivate/FH/data/fccd_data_patient_task0_cv_$i --net CNN224_CROP --workers 6 --it 5 --epochs 200 --batch-size 32 --blur-heatmaps --objective hsc --logdir-suffix task0_no_aug_200_hsc_$i --gpu ${gpus[$i]}" C-m
    echo "TMUX session $sessname$i running."
done