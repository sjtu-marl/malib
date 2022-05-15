def hard_update(target, source):
    for t_para, s_para in zip(target.parameters(), source.parameters()):
        t_para.data.copy_(s_para.data)


def soft_update(target, source, rho):
    for t_para, s_para in zip(target.parameters(), source.parameters()):
        t_para.data.copy_(t_para.data * rho + s_para.data * (1 - rho))
