import matplotlib.pyplot as plt

# Training and evaluation ROC AUCs
# 200, 400, 600, 800, 1000 trees
st50mu  = [0.978, 0.968, 0.981, 0.967, 0.983, 0.967, 0.983, 0.967, 0.984, 0.967]
st75mu  = [0.980, 0.975, 0.983, 0.976, 0.985, 0.975, 0.986, 0.975, 0.987, 0.975]
st100mu = [0.978, 0.978, 0.982, 0.979, 0.984, 0.979, 0.985, 0.979, 0.986, 0.979]
st200mu = [0.975, 0.974, 0.980, 0.976, 0.982, 0.977, 0.983, 0.977, 0.985, 0.977]
st400mu = [0.965, 0.957, 0.969, 0.958, 0.972, 0.957, 0.975, 0.958, 0.976, 0.957]

st50mu_t  = [st50mu[0], st50mu[2], st50mu[4], st50mu[6], st50mu[8]]
st50mu_e  = [st50mu[1], st50mu[3], st50mu[5], st50mu[7], st50mu[9]]
st75mu_t  = [st75mu[0], st75mu[2], st75mu[4], st75mu[6], st75mu[8]]
st75mu_e  = [st75mu[1], st75mu[3], st75mu[5], st75mu[7], st75mu[9]]
st100mu_t = [st100mu[0], st100mu[2], st100mu[4], st100mu[6], st100mu[8]]
st100mu_e = [st100mu[1], st100mu[3], st100mu[5], st100mu[7], st100mu[9]]
st200mu_t = [st200mu[0], st200mu[2], st200mu[4], st200mu[6], st200mu[8]]
st200mu_e = [st200mu[1], st200mu[3], st200mu[5], st200mu[7], st200mu[9]]
st400mu_t = [st400mu[0], st400mu[2], st400mu[4], st400mu[6], st400mu[8]]
st400mu_e = [st400mu[1], st400mu[3], st400mu[5], st400mu[7], st400mu[9]]

n_trees = [200, 400, 600, 800, 1000]

fig, ax = plt.subplots()
ax.plot(n_trees, st50mu_t, 'b-', label="50 microns")
ax.plot(n_trees, st75mu_t, 'g-', label="75 microns")
ax.plot(n_trees, st100mu_t, 'r-', label="100 microns")
ax.plot(n_trees, st200mu_t, 'c-', label="200 microns")
ax.plot(n_trees, st400mu_t, 'm-', label="400 microns")

ax.set_xlabel("Number of trees in TMVA forest")
ax.set_ylabel("ROC AUC")
ax.legend()
ax.set_title("ROC AUCs for Training Data")

plt.savefig("roc-auc-training.png")

fig, ax = plt.subplots()
ax.plot(n_trees, st50mu_e, 'b-', label="50 microns")
ax.plot(n_trees, st75mu_e, 'g-', label="75 microns")
ax.plot(n_trees, st100mu_e, 'r-', label="100 microns")
ax.plot(n_trees, st200mu_e, 'c-', label="200 microns")
ax.plot(n_trees, st400mu_e, 'm-', label="400 microns")

ax.set_xlabel("Number of trees in TMVA forest")
ax.set_ylabel("ROC AUC")
ax.legend()
ax.set_title("ROC AUCs for Evaluation Data")

plt.savefig("roc-auc-evaluation.png")
