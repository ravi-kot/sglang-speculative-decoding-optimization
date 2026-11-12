echo $$ > ~/lab/runq.pid
cd ~/lab/env/q36
mkdir -p ~/lab/profiles/q36_ft2
for it in "$@"; do echo "$(date -u +%T) start $it" >> ~/lab/profiles/q36_ft2/runq.log; bash run17.sh $it > ~/lab/profiles/q36_ft2/$it.out 2>&1; echo "$(date -u +%T) end $it rc=$?" >> ~/lab/profiles/q36_ft2/runq.log; done
rm -f ~/lab/runq.pid
