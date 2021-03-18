from pyqubo import Binary
from pprint import pprint
import requests
import json
import re

#  Fujitsu DA acccess key
# ---------------------------https://api.aispf.global.fujitsu.com/da/v2/async/qubo/solve 
post_url = 'https://api.aispf.global.fujitsu.com/da/v2/qubo/solve'
post_headers = {'X-Api-Key' : 'aac938c03db1254b7b1cef3f6961eb9bf101ade2fe8aaa4b3b322d5d11f43793', \
                   'Accept': 'application/json', \
                   'Content-Type': 'application/json'}
# ---------------------------

def req(ddic):
    #POSTパラメータは二つ目の引数に辞書で指定する 
    response = requests.post(post_url,
            json.dumps(ddic), \
            headers=post_headers)
    print(response.json())
    print(type(response.json()))
    return response

def qubodic_to_fdic(qdic, offset):
    # pyqubo x0, x1 to Fujitsu DA BinaryPolynomial
    fdic = {}
    gdic = {}
    alist = []
    for k, v in qdic.items():
        edic = {}
        a0 = re.sub(r'^[a-zAz]', '', k[0])
        a0 = re.sub(r'$', '', a0)
        a1 = re.sub(r'^[a-zAz]', '', k[1])
        a1 = re.sub(r'$', '', a1)
        edic["coefficient"] = v
        edic["polynomials"] = [int(a0), int(a1)]
        alist.append(edic)
    edic = {}
    edic["coefficient"] = offset
    alist.append(edic)
    gdic["terms"] = alist
    fdic["binary_polynomial"] = gdic 

    # ddicには fujitsuDAPTのパラメータを追加
    # ddic ={"solution_mode":  "QUICK", \
      #         "number_iterations": 100, \
       #        }
    ddic = {"solution_mode" : "COMPLETE", "number_iterations": 1000000, "number_runs": 20, "offset_increase_rate": 819, "temperature_start": 655, "temperature_decay": 0.0001, "temperature_mode": "EXPONENTIAL", "temperature_interval": 100, "noise_model": "METROPOLIS"}
    fdic["fujitsuDA2PT"] = ddic #fujitsuDAPT, fujitsuDA2PT などがある。
    return fdic

# ハミルトニアンを記述　4個のうち1つを1にする
q1, q2, q3, q4 = Binary("q1"), Binary("q2"), Binary("q3"), Binary("q4")
H = (q1+q2+q3+q4-1)**2 


# Create QUBO
model = H.compile()
qubo, offset = model.to_qubo()

pprint(qubo)  ## QUBO形式の表示用
print(offset)
print("####################")

# Solve QUBO model by Fujitsu DA
# ---------------------------

fdic = qubodic_to_fdic(qubo, offset)  
pprint(fdic) #FujitsuデジタルアニーラのQUBO形式を確認
print("####################")
resdic = req(fdic) #jsonを送信
