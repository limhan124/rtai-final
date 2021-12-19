import filecmp
import subprocess
import os
import time

baseFilePath = "/cluster/home/limhan/release_21"
not_verified = True

id = 0
scores = []
total_score = 0
res_list = []
false_res = []
pre_list = ["/test_cases", "/prelim_test_cases"]
# pre_list = ["/test_cases"]
for i in range(2):
    pre = pre_list[i]
    f = open(baseFilePath + "{}/gt.txt".format(pre), "r")
    lines = f.readlines()
    # print(lines)
    for line in lines:
        if not line.startswith("net"):
            continue
        if line.startswith("net,test case,"):
            continue
        a = line.split(",")
        net = a[0]
        img = a[1]
        truth = a[2].replace("\n", "")
        if truth == "verified":
            total_score += 1
        if not not_verified:
            if truth == "not verified":
                continue
        time_0 = time.time()
        cmd = "python verifier.py --net {} --spec ..{}/{}/{}".format(net, pre, net, img)
        print(cmd)
        run_res = subprocess.Popen(cmd, shell=True, encoding="utf-8", stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, cwd=baseFilePath + "/code")
        try:
            run_res.wait(60)
            # vrf = run_res.communicate()[0].split("\n")[-2].replace("\n", "")
            vrf = run_res.communicate()[0].replace("\n", "")
        except subprocess.TimeoutExpired:
            vrf = "not verified"
            run_res.kill()
        time_1 = time.time()
        score = 1 if vrf == "verified" and truth == "verified" else -2 if vrf == "verified" and truth == "not verified" \
                  else 0
        scores.append(score)
        res = "{},{},{},{}(gt),{}(out)".format(id, net, img, truth, vrf) \
              + "\t\tRunning time: %.3f" % (time_1 - time_0)
        print(res)
        id += 1
        res_list.append(res)
        if vrf != truth:
            false_res.append(res)
res_list.append("true score: " + str(sum(scores)) + "/" + str(total_score))

with open(baseFilePath + "/code/" + "res_{}.txt".format("v" if not not_verified else "v_nv"), 'w') as f:
    for i in res_list:
        f.write(i + "\n")
    f.write('\n\n')
    for i in false_res:
        f.write(i + "\n")
    f.close()