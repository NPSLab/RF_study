rm -f et*.cpp

cp start_template.cpp et.cpp
echo "tra_sub_node(0)" >> et.cpp
g++ -E et.cpp -o et1.cpp

cp start_template.cpp et.cpp
cat et1.cpp >> et.cpp
g++ -E et.cpp -o et2.cpp

cp start_template.cpp et.cpp
cat et2.cpp >> et.cpp
g++ -E et.cpp -o et3.cpp

cp end_template.cpp et.cpp
cat et3.cpp >> et.cpp
g++ -E et.cpp -o tmp.inc

sed '/^#/d' < tmp.inc > et3.inc

rm tmp.inc
rm et*.cpp
