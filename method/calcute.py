from sympy import symbols, Eq, solve


def svd_compress_ratio(rank):
    # 定义变量
    ratio = symbols('x')
    
    # 定义方程
    equation = Eq(rank*(1024+2048)*3*64*16+476710912, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, ratio)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def have_ratio_to_solve_svd_rank(ratio):
    # 定义变量
    rank = symbols('x')
    
    # 定义方程
    equation = Eq(rank*(1024+2048)*3*64*16+476710912, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, rank)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def have_ratio_and_pruning_num_to_solve_svd_rank(ratio, num_pruning):
    # 定义变量
    rank = symbols('x')
    
    # 定义方程
    equation = Eq(rank*(1024+2048)*3*(64*16-num_pruning)+476710912, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, rank)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal


def pruning_compress_ratio(num_pruning):
    # 定义变量
    ratio = symbols('x')
    
    # 定义方程
    equation = Eq(476710912+(16*64-num_pruning)*1024*2048*3, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, ratio)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def have_ratio_to_solve_pruning_num(ratio):
    # 定义变量
    num_pruning = symbols('x')
    
    # 定义方程
    equation = Eq(476710912+(16*64-num_pruning)*1024*2048*3, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, num_pruning)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def have_merging_group_num_and_ratio_to_solve_pruning_num(num_group, ratio):
    # 定义变量
    num_pruning = symbols('x')
    
    # 定义方程
    equation = Eq(476710912+(num_group-num_pruning)*1024*2048*3, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, num_pruning)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def have_ratio_to_solve_merging_num_group(ratio):
    # 定义变量
    num_group = symbols('x')
    
    # 定义方程
    equation = Eq(476710912+(num_group)*1024*2048*3, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, num_group)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def have_ratio_and_merging_group_num_to_solve_svd_rank(ratio, num_group):
    # 定义变量
    rank = symbols('x')
    
    # 定义方程
    equation = Eq(rank*(1024+2048)*3*(num_group)+476710912, (1-ratio)*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, rank)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal


# 单独做pruning的时候，通过压缩率，计算出剪枝的专家个数
def pruning_num_cal(ratio):
    _, result = have_ratio_to_solve_pruning_num(ratio)
    return round(result[0])

def svd_rank_cal(ratio):
    _, result =  have_ratio_to_solve_svd_rank(ratio)
    #print(f"compress ratio:{ratio}  =====>  SVD rank:{result}")
    return round(result[0])
def pruning_and_svd_compress_ratio(rank):
    # 定义变量
    ratio = symbols('x')
    
    # 定义方程
    equation = Eq(rank*(1024+2048)*3*64*16+476710912, ratio*(1024*2048*3*64*16+476710912))
    
    # 解方程
    solution = solve(equation, ratio)
    solution_decimal = [sol.evalf() for sol in solution]
    
    return solution, solution_decimal

def pruning_svd(all_ratio, pruning_ratio):
    _, result = have_ratio_to_solve_pruning_num(pruning_ratio * all_ratio)
    num_experts_pruning = round(result[0])  # 四舍五入
    # 已知总的压缩率，以及剪枝的承担的压缩比例，和剪枝专家个数，求 SVD 的 rank
    _, result = have_ratio_and_pruning_num_to_solve_svd_rank(all_ratio, num_experts_pruning)
    lora_rank = round(result[0])  # 四舍五入

    return num_experts_pruning, lora_rank

def pruning_merging_num_cal(all_ratio, pruning_ratio):
    _, result = have_ratio_to_solve_pruning_num(pruning_ratio * all_ratio)
    num_experts_pruning = round(result[0])  # 四舍五入
    # 已知总的压缩率，以及剪枝的承担的压缩比例，和剪枝专家个数，求 SVD 的 rank
    _, result = have_ratio_to_solve_merging_num_group(all_ratio)
    num_group = round(result[0])  # 四舍五入

    return num_experts_pruning, num_group

def merging_pruning_num_cal(all_ratio, pruning_ratio):
    _, result = have_ratio_to_solve_merging_num_group((1-pruning_ratio) * all_ratio)
    num_group = round(result[0])  # 四舍五入
    _, result = have_merging_group_num_and_ratio_to_solve_pruning_num(num_group, all_ratio)
    num_experts_pruning = round(result[0])  # 四舍五入
    # 已知总的压缩率，以及剪枝的承担的压缩比例，和剪枝专家个数，求 SVD 的 rank
    
    return num_experts_pruning, num_group


def merging_only_num_cal(all_ratio):
    _, result = have_ratio_to_solve_merging_num_group(all_ratio)
    num_group = round(result[0])  # 四舍五入
    
    return num_group

def merging_svd_cal(all_ratio, merging_ratio):
    _, result = have_ratio_to_solve_merging_num_group(merging_ratio * all_ratio)
    num_group = round(result[0])  # 四舍五入

    _, result = have_ratio_and_merging_group_num_to_solve_svd_rank(all_ratio, num_group)
    lora_rank = round(result[0])  # 四舍五入

    return lora_rank, num_group


def pruning_merging_svd_cal(all_ratio, pruning_ratio, merging_ratio):
    _, result = have_ratio_to_solve_pruning_num(pruning_ratio * all_ratio)
    num_experts_pruning = round(result[0])  # 四舍五入
    # 已知总的压缩率，以及剪枝的承担的压缩比例，和剪枝专家个数，求 SVD 的 rank
    _, result = have_ratio_to_solve_merging_num_group(all_ratio*merging_ratio+pruning_ratio * all_ratio)
    num_group = round(result[0])  # 四舍五入
    
    _, result = have_ratio_and_merging_group_num_to_solve_svd_rank(all_ratio, num_group)
    lora_rank = round(result[0])  # 四舍五入

    return num_experts_pruning, num_group, lora_rank



