total_ranks=4
mode=$1 # eval / sample

for ((rank=0; rank<total_ranks; rank++))
do
    sbatch /ailab/user/wangliuyi/code/w61_grutopia/scripts/slurm/one_slurm.sh $mode $rank
done

echo "All jobs submitted"