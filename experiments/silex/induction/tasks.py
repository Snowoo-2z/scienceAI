"""S7 internally authored protocol, fixed before first run.
Programs are evaluator-only; the learner receives only initial I/O pairs.
Query pools provide inputs without labels. Teachers are simulated and exact.
"""
POOLS={
 'int':[-9,-5,-3,-2,-1,0,1,2,3,4,5,8,11],
 'text':['  A B  ','B_A','a,b,C','ÉTÉ  hiver',' z ','XYZ','b a b','a a','  7  8 ', 'AbCd', 'a,,b'],
 'list':[[3,1,2],[-4,1,-2],[2,2,1],[],[0],[5,-5],[4,3,2,1],[-1,-1,2],[9,0,3]],
}
TEST={
 'int':[-17,-7,6,9,13,22,31],
 'text':['  BONJOUR MONDE  ','Un_deux_TROIS','c,b,a','éTé HIVER','x y x','  Alpha   beta ', 'six,sept,HUIT'],
 'list':[[-8,5,2,5],[10,-3,0],[-5,-4,-3],[8,2,8,1],[12],[-7,7,9,0],[4,4,4,2]],
}
# Includes an intentionally unrepresentable 4-step task at search depth 3.
TASKS=[
 ('n_double_plus_one','int',['mul2','add1'],[1]),
 ('n_absolute','int',['absolute'],[1]),
 ('n_square_plus_two','int',['square','add2'],[0]),
 ('n_negative_triple','int',['mul3','negate'],[2]),
 ('n_double_square','int',['square','mul2'],[1]),
 ('t_clean_lower','text',['strip','lower'],[' ABC ']),
 ('t_words_reverse','text',['split_space','reverse','join_space'],['a b']),
 ('t_comma_sort','text',['split_comma','sort'],['b,a']),
 ('t_first_two_lower','text',['lower','take2'],['ABC']),
 ('t_words_count','text',['split_space','length'],['a b']),
 ('l_positive_sum','list',['positive_only','sum'],[[1,2]]),
 ('l_unique_sorted','list',['unique','sort'],[[2,1,2]]),
 ('l_abs_sorted','list',['abs_each','sort'],[[-2,1]]),
 ('l_reverse_take','list',['reverse','take2'],[[1,2]]),
 ('l_sum_plus_one','list',['sum','add1'],[[1,2]]),
 ('outside_depth','text',['strip','lower','split_comma','sort'],[' B,a ']),
]
