test = [
  [1,3,1],
  [1,5,1],
  [4,2,1]
]

grid = [[1,3,1]]

grid = [
  [1,3,1],
  [1,5,1],
  [4,2,1]
]

grid = [[1]]
top_grid = grid[:-1]
left_grid = [line[:-1] for line in grid]

print(top_grid)
print(left_grid)