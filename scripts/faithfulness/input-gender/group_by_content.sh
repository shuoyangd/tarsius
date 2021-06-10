# $1 prefx
# $2 saliency
# $3 dir

head -500 $1 > prefx_hf1
tail -500 $1 > prefx_hf2
head -500 $2 > prefx_sa1
tail -500 $2 > prefx_sa2
idx=0

python delete_words.py < prefx_hf1 | sort | uniq > tmp_templates
paste prefx_hf1 prefx_sa1 > tmp_concat
mkdir -p $3

while IFS= read -r line
do
  field2=`echo $line | cut -d',' -f 2`
  field3=`echo $line | cut -d',' -f 3`
  cat tmp_concat | grep "$field2" | grep "$field3" | cut -f 2 > $3/$idx.sa
  cat tmp_concat | grep "$field2" | grep "$field3" | cut -f 1 > $3/$idx.prefx
  idx=$(( idx + 1))
done < tmp_templates

python delete_words.py < prefx_hf2 | sort | uniq > tmp_templates
paste prefx_hf2 prefx_sa2 > tmp_concat

while IFS= read -r line
do
  field2=`echo $line | cut -d',' -f 2`
  field3=`echo $line | cut -d',' -f 3`
  cat tmp_concat | grep "$field2" | grep "$field3" | cut -f 2 > $3/$idx.sa
  cat tmp_concat | grep "$field2" | grep "$field3" | cut -f 1 > $3/$idx.prefx
  idx=$(( idx + 1))
done < tmp_templates

rm tmp_templates
rm tmp_concat
