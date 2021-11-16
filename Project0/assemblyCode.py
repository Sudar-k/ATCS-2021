import dis

def addTwo():
    x = 10
    y = 15
    z = x + y
    return z

dis.dis(addTwo)

bytecode = dis.Bytecode(addTwo)
for i in bytecode:
    print(i.opname)