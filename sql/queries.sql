-- Q1: Completed orders with spend > 500, customer total > 8000
SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2: Revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city ASC, revenue DESC;

-- Q3: Products above average catalogue price
SELECT product_id, product_name, category, ROUND(price_inr, 2) AS price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: Customers with zero completed orders
SELECT c.customer_id, c.city, c.signup_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id ASC;

-- Q5: Customers spending above average customer total spend
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
overall_avg AS (
    SELECT AVG(total_spend) AS avg_spend FROM customer_totals
)
SELECT ct.customer_id,
       ROUND(ct.total_spend, 2) AS total_spend,
       ROUND(oa.avg_spend, 2) AS avg_benchmark
FROM customer_totals ct
CROSS JOIN overall_avg oa
WHERE ct.total_spend > oa.avg_spend
ORDER BY total_spend DESC;

-- Q6: Window function rankings by city
WITH city_customer_spend AS (
    SELECT c.city, o.customer_id, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city, customer_id, total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM city_customer_spend
ORDER BY city ASC, total_spend DESC;

-- Q7: Running cumulative completed revenue
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS order_date, ROUND(SUM(order_amount), 2) AS daily_rev
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT order_date, daily_rev,
    ROUND(SUM(daily_rev) OVER (ORDER BY order_date ASC), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date ASC;
