import json

with open('profile-ss-csr-pref-ains-45.json', 'r') as f, open('missing-pref-ains-1.json', 'r') as m1, open('missing-pref-ains-2.json', 'r') as m2:
    old = json.load(f)
    m1_ = json.load(m1)
    m2_ = json.load(m2)

m1__ = {e["args"]: e for e in m1_}
m2__ = {e["args"]: e for e in m2_}
new = []
missing = []

for e in old:
    if e["status"] == "initializing":
        try:
            tmp = m1__[e["args"]]
            assert tmp["status"] != "initializing"
            new.append(tmp)
        except KeyError:
            try:
                tmp = m2__[e["args"]]
                assert tmp["status"] != "initializing"
                new.append(tmp)
            except KeyError:
                missing.append(e["args"])
    else:
        new.append(e)

with open('profile.json', 'w') as f:
    f.write(json.dumps(new, indent=4))

with open('experiment-2.sh', 'w') as f:
    f.writelines(f"python {item}\n" for item in missing)


