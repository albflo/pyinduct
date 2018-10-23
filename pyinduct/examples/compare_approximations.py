import pickle
import pyinduct as pi

with open("diff_sys.pkl", "rb") as f:
    data = pickle.load(f)

with open("diff_sys_approx.pkl", "rb") as f:
    data_approx = pickle.load(f)


diff = [d - a for d, a in zip(data, data_approx)]

for d in diff:
    print(d.output_data.min())
    print(d.output_data.max())

pi.PgSurfacePlot(diff)
pi.show()
