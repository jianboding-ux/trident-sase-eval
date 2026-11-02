#!/bin/bash
# Downloads the binaries and the public datasets used by TRIDENT.
set -e
cd "$(dirname "$0")/.."
mkdir -p bin data/unsw data/nslkdd data/cicids2017
curl -sL -o bin/opa https://github.com/open-policy-agent/opa/releases/download/v1.22.0/opa_linux_amd64_static && chmod +x bin/opa
curl -sL https://github.com/tsenart/vegeta/releases/download/v12.12.0/vegeta_12.12.0_linux_amd64.tar.gz | tar xz -C bin vegeta
# UNSW-NB15 official partitions (Moustafa & Slay, 2015) and NSL-KDD (Tavallaee et al., 2009); CIC-IDS2017 (Sharafaldin et al., 2018)
curl -sL -o data/unsw/UNSW_NB15_training-set.csv https://raw.githubusercontent.com/ushukkla/nospammers/master/UNSW_NB15_training-set.csv
curl -sL -o data/unsw/UNSW_NB15_testing-set.csv  https://raw.githubusercontent.com/ushukkla/nospammers/master/UNSW_NB15_testing-set.csv
curl -sL -o "data/nslkdd/KDDTrain+.txt" "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
curl -sL -o "data/nslkdd/KDDTest+.txt"  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
B=https://raw.githubusercontent.com/HY21797/CICIDS2017-models/main/MachineLearningCVE
for f in Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \
         Friday-WorkingHours-Morning.pcap_ISCX.csv Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv; do
  curl -sL -o data/cicids2017/ML_$f "$B/$f"; done
curl -sL -o data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
  https://raw.githubusercontent.com/SHUBHAMRAWAT-25/DDOS-attack-detection-using-random-forest-algorithm/main/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
echo "done"
