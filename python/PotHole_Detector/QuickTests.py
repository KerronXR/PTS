# Find original Ps:
p1_ori = 0.8
p2_ori = 0.8
p3_ori = 0.8

q1_ori = 1 - p1_ori
q2_ori = 1 - p2_ori
q3_ori = 1 - p3_ori

ps = p1_ori * p2_ori * q3_ori + p1_ori * p3_ori * q2_ori + p2_ori * p3_ori * q1_ori + p1_ori * p2_ori * p3_ori
print(">ps (3 members): " + str(ps) + ', 1st: ' + str(p1_ori) + ', 2nd: ' + str(p1_ori) + ', 3rd: ' + str(p1_ori))
# 5345345353667501
n = 5345345353667501
added = 2
k = n + added

q = ((2 * k + 2) / (2 * k + 3) * ps + (1 / (2 * (2 * k + 3))))
print('Is not efficient to enlarge if Q(expected) is less than: ' + str(q))

q_right = ((n + 1) / (n + 2)) * ps + (1 / (2 * (n + 1)))
print('from formula - Q = ' + str(q_right))
if q_right < q:
    print(str(q_right) + ' < ' + str(q) + ' -> Not efficient to enlarge the team')
else:
    print(str(q_right) + ' > ' + str(q) + ' -> ENLARGE THE TEAM!!11')

delta = added*(ps - q)
print('Delta: ' + str(delta))

p1 = p1_ori - delta
p2 = p2_ori - delta
p3 = p3_ori - delta
p4 = 0.5 + delta
p5 = 0.5 + delta

# p1 = 0.8
# p2 = 0.8
# p3 = 0.8
# p4 = 0.65
# p5 = 0.65

q1 = 1 - p1
q2 = 1 - p2
q3 = 1 - p3
q4 = 1 - p4
q5 = 1 - p5

one = (
        p1 * p2 * p3 * q4 * q5 + p1 * p2 * p4 * q3 * q5 + p1 * p2 * p3 * q3 * q4 + p1 * p3 * p4 * q2 * q5 + p1 * p4 * p5 * q2 * q3 + p1 * p3 * p5 * q2 * q4 + p2 * p3 * p4 * q1 * q5 + p2 * p3 * p5 * q1 * q4 + p2 * p4 * p5 * q1 * q3 + p3 * p4 * p5 * q1 * q2)
two = (
        p1 * p2 * p3 * p4 * q5 + p1 * p2 * p3 * p5 * q4 + p1 * p2 * p4 * p5 * q3 + p1 * p3 * p4 * p5 * q2 + p2 * p3 * p4 * p5 * q1)
there = (p1 * p2 * p3 * p4 * p5)
pt = one + two + there
print(">pt (5 members): " + str(pt)
      + ', 1st: ' + str(p1)
      + ', 2nd: ' + str(p2)
      + ', 3rd: ' + str(p3)
      + ', 4th: ' + str(p4)
      + ', 5th: ' + str(p5))
