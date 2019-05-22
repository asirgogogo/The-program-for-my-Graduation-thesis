import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def figure(host,f,list1=None,list2=None):
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()

    offset = 0
    new_fixed_axis = par1.get_grid_helper().new_fixed_axis
    par1.axis["right"] = new_fixed_axis(loc="right",axes=par1,offset=(offset, 0))
    par1.axis["right"].toggle(all=True)

    offset = 40
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par2.axis["right"] = new_fixed_axis(loc="right",axes=par2,offset=(offset, 0))
    par2.axis["right"].toggle(all=True)

    offset = 80
    new_fixed_axis = par3.get_grid_helper().new_fixed_axis
    par3.axis["right"] = new_fixed_axis(loc="right",axes=par3, offset=(offset, 0))
    par3.axis["right"].toggle(all=True)

    offset = 120
    new_fixed_axis = par4.get_grid_helper().new_fixed_axis
    par4.axis["right"] = new_fixed_axis(loc="right",axes=par4, offset=(offset, 0))
    par4.axis["right"].toggle(all=True)
    if list1 != None or list2 != None:
        offset = 180
        new_fixed_axis = par4.get_grid_helper().new_fixed_axis
        par5.axis["right"] = new_fixed_axis(loc="right", axes=par5, offset=(offset, 0))
        par5.axis["right"].toggle(all=True)

    host.set_ylim(0,100)
    host.set_xlabel("索引")
    host.set_ylabel("推进速度")
    par1.set_ylabel("刀盘扭矩")
    par2.set_ylabel("刀盘转速")
    par3.set_ylabel("螺机转速")
    par4.set_ylabel("总推进力")
    if list1!=None or list2!=None:
        par5.set_ylabel("分割线")

    n = f["TJSD"].values.shape[0]
    l = [i + 1 for i in range(n)]
    p1= host.plot(l,f["TJSD"])
    p2= par1.plot(l,f["DPNJ"])
    p3= par2.plot(l,f["DPZS"])
    p4= par3.plot(l,f["LJZS"])
    p5=par4.plot(l,f["ZTJL"])

    if list1 != None:
        for i in range(len(list1)):
            par5.plot([list1[i],list1[i]],[0,30000],c="black",linestyle="-")
    if list2 != None:
        for j in range(len(list2)):
            par5.plot([list2[j], list2[j]], [0, 30000], c="black", linestyle=":")
    host.legend(loc="upper left")