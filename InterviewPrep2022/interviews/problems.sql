# 176
#  Write an SQL query to report the second highest salary from the Employee table. If there is no second highest salary, the query should report null

SELECT 
IFNULL(
    
(SELECT DISTINCT
salary
FROM employee
ORDER BY salary DESC
LIMIT 1 OFFSET 1), NULL) AS SecondHighestSalary

# 177
# Write an SQL query to report the nth highest salary from  the Employee table. If there is no nth highest salary, the query should report null.
 

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    SET N=N-1;
  RETURN (
      SELECT DISTINCT Salary FROM Employee ORDER BY Salary DESC LIMIT N,1
  );
END


# 181
#Write an SQL query to find the employees who earn more than their managers.

#Return the result table in any order.


SELECT a.Name as 'Employee'
FROM 
Employee as a JOIN
Employee as b
WHERE 
a.ManagerId = b.Id
AND a.Salary > b.Salary

# 184
# Write an SQL query to find employees who have the highest salary in each of the departments.


SELECT
b.name as Department
,a.name as Employee
,a.salary as Salary

FROM
Employee as a
LEFT JOIN
Department as b
on a.departmentId = b.id

WHERE
(a.departmentId, a.salary) IN
(SELECT
departmentId, Max(salary)
 FROM 
 Employee
 GROUP BY departmentId
)

# 262
# The cancellation rate is computed by dividing the number of canceled (by client or driver) requests with unbanned users by the total number of requests with unbanned users on that day.


select t.Request_at Day,
       ROUND((count(IF(t.status!='completed',TRUE,null))/count(*)),2) as 'Cancellation Rate'
from Trips t where 
t.Client_Id in (Select Users_Id from Users where Banned='No') 
and t.Driver_Id in (Select Users_Id from Users where Banned='No')
and t.Request_at between 

# 180
# Write an SQL query to find all numbers that appear at least three times consecutively.
SELECT DISTINCT
    l1.Num AS ConsecutiveNums
FROM
    Logs l1,
    Logs l2,
    Logs l3
WHERE
    l1.Id = l2.Id - 1
    AND l2.Id = l3.Id - 1
    AND l1.Num = l2.Num
    AND l2.Num = l3.Num
;



# 185
#A company's executives are interested in seeing who earns the most money in each of the company's departments. A high earner in a department is an employee who has a salary in the top three unique salaries for that department.

SELECT  d.Name AS Department, e.Name AS Employee , e.Salary 
FROM Employee AS e, Employee as e1, Department AS d
WHERE e.DepartmentId = d.Id
AND e1.DepartmentId = e.DepartmentId
AND e1.Salary >= e.Salary 
GROUP BY e.Id
HAVING COUNT(DISTINCT e1.Salary) <= 3;