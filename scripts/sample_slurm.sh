total_ranks=4

for ((rank=0; rank<total_ranks; rank++))
do
    sbatch /ailab/user/wangliuyi/code/w61_grutopia/scripts/sample_one_slurm.sh $rank
done

echo "All jobs submitted"