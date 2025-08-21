1- https://leetcode.com/problems/minimum-difficulty-of-a-job-schedule/description/
----------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int minDifficulty(int[] dif, int d) {
        int n = dif.length;
        if(n<d ) return -1;
        int dp[][]= new int [n+1][d+1];
        for(int i=0;i<=n;i++) Arrays.fill(dp[i],-1);
        return solve(dif,d,0,dp);
    }
    public int solve(int []dif, int days, int idx, int dp[][]){
        if(days==1){
            int max =0;
            for(int i=idx;i<dif.length;i++) max =Math.max(max,dif[i]);
            return max;
        }
        if(dp[idx][days]!= -1) return dp[idx][days];
        int max =0;
        int ans = Integer.MAX_VALUE;
        for(int i =idx;i<=dif.length-days;i++){
            max = Math.max(max,dif[i]);
            ans = Math.min(ans,max + solve(dif,days-1,i+1,dp));
        }
        return dp[idx][days]=ans;
    }
}

//Approach-1 (Recursion + Memoization)
//T.C : O(n^2*d)
//S.C : O(301*11) ~= O(1)
-----------------------------------------------------------------------------------------------------------------------------------------------------
2- https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int maxLength(List<String> arr) {
        return findMaxLen(arr,0,"");
    }

    public int findMaxLen(List<String> arr, int idx, String temp){
        if(idx >=arr.size()) return temp.length();
        int include =0,exclude =0;
        if(isValid(temp,arr.get(idx))){
            include = findMaxLen(arr, idx + 1, temp + arr.get(idx));
        }
        exclude = findMaxLen(arr, idx+1,temp);
        return Math.max(include, exclude);
        
    }

    public boolean isValid(String s1, String s2){
        int []arr = new int[26];
        for (char c : s1.toCharArray()) {
            if (arr[c - 'a']++ > 0) return false;
        }
        for (char c : s2.toCharArray()) {
            if (arr[c - 'a']++ > 0) return false;
        }
        return true; 
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
3- https://leetcode.com/problems/ugly-number-ii/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int nthUglyNumber(int n) {
        int ans[] = new int[n+1];
        int n1=1,n2=1,n3=1;
        ans[1]=1;
        for(int i=2;i<=n;i++){
            int i2th= ans[n1] *2;
            int i3th= ans[n2] *3;
            int i4th= ans[n3] *5;

            int minNum = Math.min(i2th,Math.min(i3th,i4th));
            ans[i] = minNum;

            if(i2th == minNum) n1++;
            if(i3th == minNum) n2++;
            if(i4th == minNum) n3++;
        }
        return ans[n];
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
4- https://leetcode.com/problems/perfect-squares/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int numSquares(int n) {
        int dp[] = new int[n+1];
        Arrays.fill(dp, -1);
        return solve(n,dp);
    }
    public int solve(int n, int []dp){
        if(n==0) return 0;
        if(dp[n]!=-1) return dp[n];
        int ans = Integer.MAX_VALUE;
        for(int i=1;i<=Math.sqrt(n);i++){
            int temp = 1 + solve(n - i*i,dp);
            ans = Math.min(ans, temp);
        }
        return dp[n]= ans;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
5- https://leetcode.com/problems/maximum-profit-in-job-scheduling/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    // 1D Dp
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        int [][]arr = new int[n][3];
        int dp[] = new int[n];
        Arrays.fill(dp,-1);
        for(int i=0;i<n;i++){
            arr[i][0]= startTime[i];
            arr[i][1]= endTime[i];
            arr[i][2]= profit[i];
        }
        Arrays.sort(arr,Comparator.comparingInt(o->o[0]));
        return solve(arr, 0,n, dp);
    }
    public int solve(int [][]arr, int ind, int n, int dp[]){
        if(ind >= n) return 0;
        if(dp[ind] != -1) return dp[ind];
        int next = findNext(arr, ind+1, arr[ind][1]);
        int taken = arr[ind][2] + solve(arr, next,n, dp);
        int notTaken = solve(arr, ind+1,n, dp);
        return dp[ind] = Math.max(taken, notTaken);
    }
    public int findNext(int [][]arr, int ind, int endTimeOfPrevious){
        int l = ind, r = arr.length-1;
        int ans = arr.length+1;
        while(l<=r){
            int mid = l+(r-l)/2;
            if(arr[mid][0]>= endTimeOfPrevious){
                ans = mid;
                r= mid-1;
            }else{
                l  = mid+1;
            }
        }
        return ans;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
6- https://leetcode.com/problems/arithmetic-slices-ii-subsequence/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.*;

class Solution {
    private int solve(int[] nums, int start, int curr, int len, long diff, Map<String, Integer> dp) {
        if (curr == nums.length - 1)
            return 0;

        String key = curr + "#" + len + "#" + diff;
        if (dp.containsKey(key))
            return dp.get(key);

        int res = 0;
        for (int i = curr + 1; i < nums.length; i++) {
            long k = (long) nums[i] - (long) nums[curr];
            if (len == 1 || diff == k) {
                len++;
                if (len >= 3)
                    res++;
                res += solve(nums, start, i, len, k, dp);
                len--; // backtrack
            }
        }

        dp.put(key, res);
        return res;
    }

    public int numberOfArithmeticSlices(int[] nums) {
        int res = 0;
        Map<String, Integer> dp = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            dp.clear();
            res += solve(nums, i, i, 1, 0, dp);
        }

        return res;
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
7- https://leetcode.com/problems/climbing-stairs/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int climbStairs(int n) {
        if(n==1) return 1;
        if(n==2) return 2;
        int dp[] = new int[n+1];
        Arrays.fill(dp,-1);
        return solve(dp,n);
    }
    public int solve(int dp[], int n){
        if(n==1 || n==2) return n;
        if(dp[n]!=-1) return dp[n];
        return dp[n] = solve(dp,n-1)+solve(dp,n-2);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
8- https://leetcode.com/problems/minimum-falling-path-sum/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {

    int solve(int[][] matrix, int row, int col,int dp[][]) {
        if (row >= matrix.length || col >= matrix[0].length || col < 0) {
            return Integer.MAX_VALUE;
        }
        if (row == matrix.length - 1) {
            if (col < matrix[0].length && col >= 0)
                return matrix[row][col];
            else 
                return Integer.MAX_VALUE;
        }
        if (dp[row][col] != Integer.MIN_VALUE) {
            return dp[row][col];
        }
        return dp[row][col] = matrix[row][col] + Math.min(solve(matrix,row + 1, col,dp), 
        Math.min(solve(matrix, row + 1, col - 1,dp), solve(matrix, row + 1, col + 1,dp)));
    }

    public int minFallingPathSum(int[][] matrix) {
        int dp[][] = new int[matrix.length][matrix[0].length];
        int min_value = Integer.MAX_VALUE;
        for (int i = 0; i < matrix.length; ++i) {
            for (int [] a : dp) {
            Arrays.fill(a, Integer.MIN_VALUE);
            }
            min_value = Math.min(min_value, solve(matrix, 0, i,dp));
        }
        return min_value;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
9- https://leetcode.com/problems/house-robber/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int rob(int[] nums) {
        int dp[] = new int[nums.length+1];
        Arrays.fill(dp,-1);
        return solve(dp,nums.length-1,nums);
    }
    public int solve(int dp[], int n,int arr[]){
        if(n==0) return arr[0];
        if(n<0) return 0;
        if (dp[n] != -1) return dp[n];
        int pick = arr[n]+solve(dp,n-2,arr);
        int nonpick = solve(dp,n-1,arr);
        return dp[n] = Math.max(pick, nonpick);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
10- https://leetcode.com/problems/longest-common-subsequence/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    int[][] t;  // memoization table
    
    public int LCS(String s1, String s2, int m, int n) {
        if (m == 0 || n == 0)
            return t[m][n] = 0;
        
        if (t[m][n] != -1)
            return t[m][n];
        
        if (s1.charAt(m - 1) == s2.charAt(n - 1)) {
            return t[m][n] = 1 + LCS(s1, s2, m - 1, n - 1);
        }
        
        return t[m][n] = Math.max(LCS(s1, s2, m, n - 1),
                                  LCS(s1, s2, m - 1, n));
    }
    
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        
        // Initialize memo table with -1
        t = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                t[i][j] = -1;
            }
        }
        
        return LCS(text1, text2, m, n);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
11- https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    static int get(int[] Arr, int ind, int buy, int n, int[][] dp) {
        // Base case
        if (ind >= n) {
            return 0;
        }

        // If the result is already calculated, return it
        if (dp[ind][buy] != -1) {
            return dp[ind][buy];
        }

        int profit = 0;

        if (buy == 0) { // We can buy the stock
            profit = Math.max(0 +get(Arr, ind + 1, 0, n, dp),
                    -Arr[ind] +get(Arr, ind + 1, 1, n, dp));
        }

        if (buy == 1) { // We can sell the stock
            profit = Math.max(0 +get(Arr, ind + 1, 1, n, dp),
                    Arr[ind] +get(Arr, ind + 2, 0, n, dp));
        }

        // Store the result in dp and return it
        dp[ind][buy] = profit;
        return profit;
    }
    public int maxProfit(int[] Arr) {
        int n = Arr.length;
        int[][] dp = new int[n][2];
        
        // Initialize dp array with -1 to mark states as not calculated yet
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }

        int ans =get(Arr, 0, 0, n, dp);
        return ans;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
12- https://leetcode.com/problems/domino-and-tromino-tiling/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    final int M = 1000000007;
    int[] t = new int[1001];

    public int solve(int n) {
        if (n == 1 || n == 2)
            return n;
        if (n == 3)
            return 5;

        if (t[n] != -1)
            return t[n];

        t[n] = (int)((2L * solve(n - 1) % M + solve(n - 3)) % M);
        return t[n];
    }

    public int numTilings(int n) {
        // Initialize memoization array with -1
        for (int i = 0; i <= n; i++) {
            t[i] = -1;
        }

        return solve(n);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
13- https://leetcode.com/problems/jump-game/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    int[] t; // memoization array
    
    public boolean solve(int[] nums, int n, int idx) {
        if (idx == n - 1) 
            return true;
        
        if (t[idx] != -1) 
            return t[idx] == 1;  // convert back from int to boolean
        
        for (int i = 1; i <= nums[idx]; i++) {
            if (idx + i < n && solve(nums, n, idx + i)) {
                t[idx] = 1;  // true
                return true;
            }
        }
        
        t[idx] = 0; // false
        return false;
    }
    
    public boolean canJump(int[] nums) {
        int n = nums.length;
        t = new int[n];
        Arrays.fill(t, -1);  // initialize with -1
        return solve(nums, n, 0);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
14- https://leetcode.com/problems/flip-string-to-monotone-increasing/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    Integer[][] dp;
    public int minFlipsMonoIncr(String s) {
        int n = s.length();
        dp = new Integer[n][2]; // 0 or 1 for prev char
        return solve(s, 0, 0); // start from index 0, prev = 0
    }

    private int solve(String s, int i, int prev) {
        // Base case: reached end of string
        if (i == s.length()) return 0;

        if (dp[i][prev] != null) return dp[i][prev];

        int curr = s.charAt(i) - '0';
        int ans = Integer.MAX_VALUE;

        // Option 1: Keep current as is (only allowed if curr >= prev)
        if (curr >= prev) {
            ans = Math.min(ans, solve(s, i + 1, curr));
        }

        // Option 2: Flip current
        int flipped = 1 - curr; // flip 0→1 or 1→0
        if (flipped >= prev) {
            ans = Math.min(ans, 1 + solve(s, i + 1, flipped));
        }

        return dp[i][prev] = ans;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
15- https://leetcode.com/problems/flip-string-to-monotone-increasing/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    Integer[][] dp;
    public int minFlipsMonoIncr(String s) {
        int n = s.length();
        dp = new Integer[n][2]; // 0 or 1 for prev char
        return solve(s, 0, 0); // start from index 0, prev = 0
    }

    private int solve(String s, int i, int prev) {
        // Base case: reached end of string
        if (i == s.length()) return 0;

        if (dp[i][prev] != null) return dp[i][prev];

        int curr = s.charAt(i) - '0';
        int ans = Integer.MAX_VALUE;

        // Option 1: Keep current as is (only allowed if curr >= prev)
        if (curr >= prev) {
            ans = Math.min(ans, solve(s, i + 1, curr));
        }

        // Option 2: Flip current
        int flipped = 1 - curr; // flip 0→1 or 1→0
        if (flipped >= prev) {
            ans = Math.min(ans, 1 + solve(s, i + 1, flipped));
        }

        return dp[i][prev] = ans;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
16- https://leetcode.com/problems/minimum-path-sum/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int minPathSum(int[][] grid) {
        int dp[][]= new int[grid.length][grid[0].length];
        for(int[] arr : dp) {
            Arrays.fill(arr, -1);
        }
        return f(dp, grid, grid.length - 1, grid[0].length - 1);
    }
    
    public static int f(int dp[][], int[][] grid, int i, int j) {
        if(i == 0 && j == 0) return grid[0][0];
        if(i < 0 || j < 0) return (int)Math.pow(10, 9);
        if(dp[i][j] != -1) return dp[i][j];
        int up = grid[i][j] + f(dp, grid, i - 1, j);
        int left = grid[i][j] + f(dp, grid, i, j - 1);
        return dp[i][j] = (int)Math.min(up, left);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
17- https://leetcode.com/problems/minimum-cost-for-tickets/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    HashSet<Integer> isTravelNeeded = new HashSet<>();

    private int solve(int[] dp, int[] days, int[] costs, int currDay) {
        // If we have iterated over travel days, return 0.
        if (currDay > days[days.length - 1]) {
            return 0;
        }

        // If we don't need to travel on this day, move on to next day.
        if (!isTravelNeeded.contains(currDay)) {
            return solve(dp, days, costs, currDay + 1);
        }

        // If already calculated, return from here with the stored answer.
        if (dp[currDay] != -1) {
            return dp[currDay];
        }

        int oneDay = costs[0] + solve(dp, days, costs, currDay + 1);
        int sevenDay = costs[1] + solve(dp, days, costs, currDay + 7);
        int thirtyDay = costs[2] + solve(dp, days, costs, currDay + 30);

        // Store the cost with the minimum of the three options.
        return dp[currDay] = Math.min(oneDay, Math.min(sevenDay, thirtyDay));
    }

    public int mincostTickets(int[] days, int[] costs) {
        // The last day on which we need to travel.
        int lastDay = days[days.length - 1];
        int dp[] = new int[lastDay + 1];
        Arrays.fill(dp, -1);

        // Mark the days on which we need to travel.
        for (int day : days) {
            isTravelNeeded.add(day);
        }

        return solve(dp, days, costs, 1);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
18- https://leetcode.com/problems/reducing-dishes/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int maxSatisfaction(int[] satisfaction) {
        Arrays.sort(satisfaction); // sort to arrange properly
        int n = satisfaction.length;
        Integer[][] memo = new Integer[n][n + 1]; // dp[i][time]
        return helper(0, 1, satisfaction, memo);
    }

    private int helper(int index, int time, int[] satisfaction, Integer[][] memo) {
        if (index == satisfaction.length) return 0;

        if (memo[index][time] != null) return memo[index][time];

        // Option 1: Take this dish
        int take = satisfaction[index] * time + helper(index + 1, time + 1, satisfaction, memo);

        // Option 2: Skip this dish
        int skip = helper(index + 1, time, satisfaction, memo);

        // Store result in dp
        return memo[index][time] = Math.max(take, skip);
    }
}
//TC = sc= O(n²)
-----------------------------------------------------------------------------------------------------------------------------------------------------
19- https://leetcode.com/problems/scramble-string/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.HashMap;

class Solution {
    private HashMap<String, Boolean> memo = new HashMap<>();

    private boolean solve(String s1, String s2) {
        if (s1.equals(s2)) // includes case when both are empty
            return true;
        if (s1.length() != s2.length())
            return false;

        String key = s1 + "_" + s2;
        if (memo.containsKey(key))
            return memo.get(key);

        boolean result = false;
        int n = s1.length();

        for (int i = 1; i < n; i++) {
            // Case 1: Swapped
            boolean swapped =
                solve(s1.substring(0, i), s2.substring(n - i)) &&
                solve(s1.substring(i), s2.substring(0, n - i));

            if (swapped) {
                result = true;
                break;
            }

            // Case 2: Not Swapped
            boolean notSwapped =
                solve(s1.substring(0, i), s2.substring(0, i)) &&
                solve(s1.substring(i), s2.substring(i));

            if (notSwapped) {
                result = true;
                break;
            }
        }

        memo.put(key, result);
        return result;
    }

    public boolean isScramble(String s1, String s2) {
        memo.clear();
        return solve(s1, s2);
    }

}

-----------------------------------------------------------------------------------------------------------------------------------------------------
20- https://leetcode.com/problems/number-of-ways-of-cutting-a-pizza/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.*;

class Solution {
    int m, n;
    int[][] apples;
    int[][][] dp;
    final int MOD = 1_000_000_007;

    private int solve(int i, int j, int cutsLeft) {
        if (apples[i][j] < cutsLeft) return 0; // not enough apples
        if (cutsLeft == 1) return (apples[i][j] >= 1) ? 1 : 0;

        if (dp[i][j][cutsLeft] != -1) return dp[i][j][cutsLeft];

        long ways = 0;

        // Horizontal cuts
        for (int h = i + 1; h < m; h++) {
            if (apples[i][j] - apples[h][j] > 0) { // top has apple
                ways = (ways + solve(h, j, cutsLeft - 1)) % MOD;
            }
        }

        // Vertical cuts
        for (int v = j + 1; v < n; v++) {
            if (apples[i][j] - apples[i][v] > 0) { // left has apple
                ways = (ways + solve(i, v, cutsLeft - 1)) % MOD;
            }
        }

        return dp[i][j][cutsLeft] = (int) ways;
    }

    public int ways(String[] pizza, int k) {
        m = pizza.length;
        n = pizza[0].length();

        apples = new int[m + 1][n + 1];
        dp = new int[m + 1][n + 1][k + 1];

        for (int[][] arr2D : dp)
            for (int[] arr1D : arr2D)
                Arrays.fill(arr1D, -1);

        // Precompute apples[i][j] = number of apples in submatrix (i..m-1, j..n-1)
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                apples[i][j] = apples[i][j + 1] + apples[i + 1][j] - apples[i + 1][j + 1];
                if (pizza[i].charAt(j) == 'A') apples[i][j]++;
            }
        }

        return solve(0, 0, k);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
21- https://leetcode.com/problems/longest-palindromic-subsequence/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int longestPalindromeSubseq(String s1) {
        int n = s1.length();
        int[][] dp = new int[n + 1][n+ 1];
        for(int row[]:dp) Arrays.fill(row,-1);
        String s2 = new StringBuilder(s1).reverse().toString();
        return lcs(s1,s2,n-1,n-1,dp);
    }
    public int lcs(String s1, String s2,int i, int j,int dp[][]) {
        if(i<0 || j<0) return 0;
        if(dp[i][j] != -1) return dp[i][j];
        if(s1.charAt(i) == s2.charAt(j)) return 1 +lcs(s1,s2,i-1,j-1,dp);
        return dp[i][j]=Math.max(lcs(s1,s2,i-1,j,dp),lcs(s1,s2,i,j-1,dp));
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
22- https://leetcode.com/problems/maximum-value-of-k-coins-from-piles/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    int n;
    int[][] dp;

    private int solve(int i, List<List<Integer>> piles, int k) {
        if (i >= n) return 0;
        if (dp[i][k] != -1) return dp[i][k];

        // Not taking from this pile
        int notTaken = solve(i + 1, piles, k);

        int taken = 0, sum = 0;

        // Try taking 1, 2, ..., up to k coins from this pile
        for (int j = 0; j < Math.min(piles.get(i).size(), k); j++) {
            sum += piles.get(i).get(j);
            if (k - (j + 1) >= 0) {
                taken = Math.max(taken, sum + solve(i + 1, piles, k - (j + 1)));
            }
        }

        return dp[i][k] = Math.max(notTaken, taken);
    }

    public int maxValueOfCoins(List<List<Integer>> piles, int k) {
        n = piles.size();
        dp = new int[n][k + 1];
        for (int i = 0; i < n; i++) {
            Arrays.fill(dp[i], -1);
        }
        return solve(0, piles, k);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
23- https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/
------------------------------------------------------------------------------------------------------------------------------------------------------
//T.C : O(n*k + m*k)
//S.C : O(m*k)
class Solution {
    private int m;
    private int k;
    private final int MOD = (int) 1e9 + 7;
    private int[][] memo;

    private int solve(int i, int j, long[][] freq, String target) {
        if (i == m) {
            return 1;
        }

        if (j == k) {
            return 0;
        }

        if (memo[i][j] != -1) {
            return memo[i][j];
        }

        int notTaken = solve(i, j + 1, freq, target) % MOD;

        int taken = (int) ((freq[target.charAt(i) - 'a'][j] * solve(i + 1, j + 1, freq, target)) % MOD);

        return memo[i][j] = (notTaken + taken) % MOD;
    }

    public int numWays(String[] words, String target) {
        k = words[0].length();
        m = target.length();

        long[][] freq = new long[26][k];

        // Populate frequency array
        for (String word : words) {
            for (int col = 0; col < k; col++) {
                freq[word.charAt(col) - 'a'][col]++;
            }
        }

        memo = new int[m][k];
        for (int[] row : memo) {
            Arrays.fill(row, -1);
        }

        return solve(0, 0, freq, target);
    }
}



-----------------------------------------------------------------------------------------------------------------------------------------------------
24- https://leetcode.com/problems/number-of-ways-to-form-a-target-string-given-a-dictionary/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
//T.C : O(n*k + m*k)
//S.C : O(m*k)
class Solution {
    private int m;
    private int k;
    private final int MOD = (int) 1e9 + 7;
    private int[][] memo;

    private int solve(int i, int j, long[][] freq, String target) {
        if (i == m) {
            return 1;
        }

        if (j == k) {
            return 0;
        }

        if (memo[i][j] != -1) {
            return memo[i][j];
        }

        int notTaken = solve(i, j + 1, freq, target) % MOD;

        int taken = (int) ((freq[target.charAt(i) - 'a'][j] * solve(i + 1, j + 1, freq, target)) % MOD);

        return memo[i][j] = (notTaken + taken) % MOD;
    }

    public int numWays(String[] words, String target) {
        k = words[0].length();
        m = target.length();

        long[][] freq = new long[26][k];

        // Populate frequency array
        for (String word : words) {
            for (int col = 0; col < k; col++) {
                freq[word.charAt(col) - 'a'][col]++;
            }
        }

        memo = new int[m][k];
        for (int[] row : memo) {
            Arrays.fill(row, -1);
        }

        return solve(0, 0, freq, target);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
25- https://leetcode.com/problems/profitable-schemes/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    int mod= 1000000007;
    int[][][]memo = new int [101][101][101];
    int find(int pos,int count,int profit,int n,int minProfit,int[]group,int[]profits){
        if(pos==group.length){
            return profit>=minProfit? 1:0;
        }
        if(memo[pos][count][profit]!=-1){
            return memo[pos][count][profit];
        }
        int totalways = find(pos+1,count,profit,n,minProfit,group,profits);
        if(count+group[pos]<=n){
            totalways += find(pos+1,count+group[pos],Math.min(minProfit,profit+profits[pos]),n,minProfit,group,profits);

        }
        return memo[pos][count][profit]= totalways % mod;

    }

    public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
        for(int i=0;i<group.length;i++){
            for(int j=0;j<=n;j++){
                Arrays.fill(memo[i][j],-1);
            }
        }
        return find(0, 0, 0, n, minProfit, group, profit);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
26- https://leetcode.com/problems/restore-the-array/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
/*
    Company Tags : Google, Amazon, Microsoft, Uber
    Leetcode Link: https://leetcode.com/problems/restore-the-array/
*/

class Solution {
    
    private int n;
    private final int MOD = 1000000007;
    
    public int solve(int start, String s, int k, int[] t) {
        if(start >= n)
            return 1;
        
        if(t[start] != -1)
            return t[start];
        
        if(s.charAt(start) == '0')
            return t[start] = 0;
        
        long ans = 0;
        long num = 0;
        
        for(int end = start; end < n; end++) {
            
            num = num * 10 + (s.charAt(end) - '0');
            
            if(num > k)
                break;
            
            ans = (ans % MOD + solve(end+1, s, k, t) % MOD) % MOD;
        }
        
        return t[start] = (int)ans;
    }
    
    public int numberOfArrays(String s, int k) {
        n = s.length();
        int[] t = new int[n];
        Arrays.fill(t, -1);
        return solve(0, s, k, t);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
27- https://leetcode.com/problems/minimum-insertion-steps-to-make-a-string-palindrome/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int minInsertions(String s1) {
        int n = s1.length();
        int[][] dp = new int[n + 1][n+ 1];
        for(int row[]:dp) Arrays.fill(row,-1);
        String s2 = new StringBuilder(s1).reverse().toString();
        return n-lcs(s1,s2,n-1,n-1,dp);
    }
    public int lcs(String s1, String s2,int i, int j,int dp[][]) {
        if(i<0 || j<0) return 0;
        if(dp[i][j] != -1) return dp[i][j];
        if(s1.charAt(i) == s2.charAt(j)) return 1 +lcs(s1,s2,i-1,j-1,dp);
        return dp[i][j]=Math.max(lcs(s1,s2,i-1,j,dp),lcs(s1,s2,i,j-1,dp));
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
28- https://leetcode.com/problems/find-the-longest-valid-obstacle-course-at-each-position/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    List<Integer> answer;
    private int bisectRight(int[]A,int target,int right){
        if(right==0)
        return 0;
        int left =0;
        while(left<right){
            int mid= left+(right-left)/2;
            if(A[mid]<= target)
            left = mid+1;
            else 
            right = mid;
        }
        return left;
    }
    public int[] longestObstacleCourseAtEachPosition(int[] obstacles) {
       int n= obstacles.length,lisLength=0;
       int []answer = new int [n],lis= new int[n];
       for(int i=0;i<n;i++){
           int height = obstacles[i];
           int idx = bisectRight(lis,height,lisLength);
           if(idx== lisLength)
           lisLength++;
           lis[idx]= height;
           answer[i]= idx+1;
       }
       return answer;
       
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
29- https://leetcode.com/problems/uncrossed-lines/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    
    int m, n;
    int[][] t;
    
    private int solve(int i, int j, int[] nums1, int[] nums2) {
        
        if(i >= m || j >= n) {
            return 0;
        }
        
        if(t[i][j] != -1) {
            return t[i][j];
        }
        
        if(nums1[i] == nums2[j]) {
            return t[i][j] = 1 + solve(i+1, j+1, nums1, nums2);
        } else {
            int fix_i = solve(i, j+1, nums1, nums2);
            int fix_j = solve(i+1, j, nums1, nums2);
            return t[i][j] = Math.max(fix_i, fix_j);
        }
    }
    
    public int maxUncrossedLines(int[] nums1, int[] nums2) {
        m = nums1.length;
        n = nums2.length;
        
        t = new int[m][n];
        for(int i = 0; i < m; i++) {
            Arrays.fill(t[i], -1);
        }
        
        return solve(0, 0, nums1, nums2);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
30- https://leetcode.com/problems/solving-questions-with-brainpower/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public long mostPoints(int[][] questions) {
        int n =questions.length;
        long[]dp = new long[n];
        dp[n-1] = questions[n-1][0];
        for(int i = n-2;i>= 0;--i){
            dp[i] = questions[i][0];
            int skip = questions[i][1];
            if(i+ skip + 1 < n){
                dp[i] += dp[i +skip+1]; 
            }
            dp[i] = Math.max(dp[i],dp[i+1]);

        }
        return dp[0];
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
31- https://leetcode.com/problems/count-ways-to-build-good-strings/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int countGoodStrings(int low, int high, int zero, int one) {
        int []dp = new int[high+1];
        dp[0]=1;
        int mod = 1_000_000_007;
        for(int end =1;end<= high;++end){
            if(end>= zero){
                dp[end] += dp[end-zero];
            }
            if(end >= one){
                dp[end] += dp[end-one];
            }
            dp[end] %= mod;
        }
        int answer =0;
        for(int i= low;i<= high;++i){
            answer +=dp[i];
            answer %= mod;
        }
        return answer;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
32- https://leetcode.com/problems/count-ways-to-build-good-strings/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    int L, H, Z, O;
    int MOD = 1000000007;

    public int solve(int l, int[] dp) {
        if (l > H) return 0;

        if (dp[l] != -1) return dp[l];

        boolean addOne = false;
        if (l >= L && l <= H) {
            addOne = true;
        }

        int takeZero = solve(l + Z, dp);
        int takeOne  = solve(l + O, dp);

        return dp[l] = ( (addOne ? 1 : 0) + takeZero + takeOne ) % MOD;
    }

    public int countGoodStrings(int low, int high, int zero, int one) {
        L = low;
        H = high;
        Z = zero;
        O = one;

        int[] dp = new int[H + 1];
        Arrays.fill(dp, -1);

        return solve(0, dp);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
33- https://leetcode.com/problems/maximize-score-after-n-operations/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.*;

class Solution {
    int n;
    Map<Integer, Integer> memo;

    public int maxScore(int[] nums) {
        n = nums.length;
        memo = new HashMap<>();
        return solve(nums, 1, 0); // 0 = bitmask (no element visited yet)
    }

    private int solve(int[] nums, int operation, int mask) {
        if (memo.containsKey(mask)) {
            return memo.get(mask);
        }

        int maxScore = 0;

        for (int i = 0; i < n - 1; i++) {
            if ((mask & (1 << i)) != 0) continue; // already visited

            for (int j = i + 1; j < n; j++) {
                if ((mask & (1 << j)) != 0) continue; // already visited

                // Mark i and j as visited
                int newMask = mask | (1 << i) | (1 << j);

                int currScore = operation * gcd(nums[i], nums[j]);
                int remainingScore = solve(nums, operation + 1, newMask);

                maxScore = Math.max(maxScore, currScore + remainingScore);
            }
        }

        memo.put(mask, maxScore);
        return maxScore;
    }

    private int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
34- https://leetcode.com/problems/stone-game-ii/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
     private int f(int[] piles,int[][][] dp, int p, int i, int m){
        if(i==piles.length){
             return 0;
        }if(dp[p][i][m] != -1){
            return dp[p][i][m];
        }
        int res = p==1? 1000000:-1,s=0;
        for(int c=1;c <=Math.min(2*m,piles.length-i);c++){
            s += piles[i+c -1];
            if(p==0){// Allice
                res = Math.max(res,s+f(piles,dp,1,i+c,Math.max(m,c)));
            }else{// Bob
                res = Math.min(res,f(piles,dp,0,i+c,Math.max(m,c)));
            }
        }
            return dp[p][i][m]=res;
        
    }
    public int stoneGameII(int[] piles) {
        int[][][] dp = new int[2][piles.length+1][piles.length+1];
        for(int p=0;p<2;p++){
            for(int i=0;i<=piles.length;i++){
                for(int m=0;m<= piles.length;m++){
                    dp[p][i][m] = -1;
                }
            }
        }
        return f(piles,dp,0,0,1);
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
35- https://leetcode.com/problems/stone-game-iii/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    private int n;
    private Integer[] dp;

    private int solve(int[] stoneValue, int i) {
        if (i >= n) return 0;
        if (dp[i] != null) return dp[i];

        int take = stoneValue[i] - solve(stoneValue, i + 1);
        if (i + 1 < n) {
            take = Math.max(take, stoneValue[i] + stoneValue[i + 1] - solve(stoneValue, i + 2));
        }
        if (i + 2 < n) {
            take = Math.max(take, stoneValue[i] + stoneValue[i + 1] + stoneValue[i + 2] - solve(stoneValue, i + 3));
        }

        return dp[i] = take;
    }

    public String stoneGameIII(int[] stoneValue) {
        n = stoneValue.length;
        dp = new Integer[n];
        int diff = solve(stoneValue, 0);

        if (diff > 0) return "Alice";
        else if (diff < 0) return "Bob";
        return "Tie";
    }
}


-----------------------------------------------------------------------------------------------------------------------------------------------------
36- https://leetcode.com/problems/minimum-cost-to-cut-a-stick/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int minCost(int n, int[] cuts) {
         ArrayList<Integer> cutsList = new ArrayList<>();
        cutsList.add(0);  // Starting boundary
        for (int cut : cuts) {
            cutsList.add(cut);
        }
        cutsList.add(n);  // Ending boundary
        cutsList.sort(Integer::compareTo);  // Sort cuts
        
        int m = cutsList.size();
        int[][] dp = new int[m][m];
        for (int[] row : dp) {
            Arrays.fill(row, -1);
        }
        
        return f(1, m - 2, cutsList, dp);
    }
    public int f(int i, int j, ArrayList<Integer> cuts, int[][] dp) {
        // Base case
        if (i > j) {
            return 0;
        }

        if (dp[i][j] != -1) {
            return dp[i][j];
        }

        int mini = Integer.MAX_VALUE;

        for (int ind = i; ind <= j; ind++) {
            int ans = cuts.get(j + 1) - cuts.get(i - 1) +
                      f(i, ind - 1, cuts, dp) +
                      f(ind + 1, j, cuts, dp);

            mini = Math.min(mini, ans);
        }

        return dp[i][j] = mini;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
37- https://leetcode.com/problems/make-array-strictly-increasing/
------------------------------------------------------------------------------------------------------------------------------------------------------

import java.util.*;

class Solution {
    Map<String, Integer> memo = new HashMap<>();

    private int solve(int idx, int[] arr1, int[] arr2, int prev) {
        if (idx == arr1.length) return 0;

        String key = idx + "," + prev;
        if (memo.containsKey(key)) return memo.get(key);

        int result1 = Integer.MAX_VALUE / 2;

        // Option 1: Keep arr1[idx] if it is strictly greater than prev
        if (arr1[idx] > prev) {
            result1 = solve(idx + 1, arr1, arr2, arr1[idx]);
        }

        // Option 2: Replace arr1[idx] with some element from arr2
        int result2 = Integer.MAX_VALUE / 2;
        int i = upperBound(arr2, prev);
        if (i < arr2.length) {
            result2 = 1 + solve(idx + 1, arr1, arr2, arr2[i]);
        }

        int ans = Math.min(result1, result2);
        memo.put(key, ans);
        return ans;
    }

    public int makeArrayIncreasing(int[] arr1, int[] arr2) {
        Arrays.sort(arr2);
        memo.clear();

        int result = solve(0, arr1, arr2, Integer.MIN_VALUE);
        return result >= Integer.MAX_VALUE / 2 ? -1 : result;
    }

    // Custom upperBound function (like C++ upper_bound)
    private int upperBound(int[] arr, int target) {
        int lo = 0, hi = arr.length;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] <= target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
38- https://leetcode.com/problems/number-of-increasing-paths-in-a-grid/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    int m, n;
    int[][] directions = {
        {-1, 0}, {0, -1}, {0, 1}, {1, 0}
    };
    int[][] t;
    long MOD = (long)1e9 + 7;

    boolean isSafe(int i, int j) {
        return (i >= 0 && i < m && j >= 0 && j < n);
    }

    int dfs(int[][] grid, int i, int j) {
        if (t[i][j] != -1) 
            return t[i][j];

        long answer = 1; // path including itself

        for (int[] dir : directions) {
            int i_ = i + dir[0];
            int j_ = j + dir[1];

            if (isSafe(i_, j_) && grid[i_][j_] < grid[i][j]) {
                answer = (answer + dfs(grid, i_, j_)) % MOD;
            }
        }

        return t[i][j] = (int)answer;
    }

    public int countPaths(int[][] grid) {
        m = grid.length;
        n = grid[0].length;
        t = new int[m][n];

        // fill with -1
        for (int i = 0; i < m; i++) {
            Arrays.fill(t[i], -1);
        }

        long result = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result = (result + dfs(grid, i, j)) % MOD;
                // No of strictly oincreasing ending at (i,j)
            }
        }

        return (int)result;
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
39- https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    static int getAns(int[] Arr, int ind, int buy, int n, int fee, int[][] dp) {
        // Base case
        if (ind == n) {
            return 0;
        }

        // If the result is already calculated, return it
        if (dp[ind][buy] != -1) {
            return dp[ind][buy];
        }

        int profit = 0;

        if (buy == 0) { // We can buy the stock
            profit = Math.max(0 + getAns(Arr, ind + 1, 0, n, fee, dp), -Arr[ind] + getAns(Arr, ind + 1, 1, n, fee, dp));
        }

        if (buy == 1) { // We can sell the stock
            profit = Math.max(0 + getAns(Arr, ind + 1, 1, n, fee, dp), Arr[ind] - fee + getAns(Arr, ind + 1, 0, n, fee, dp));
        }

        // Store the result in dp and return it
        dp[ind][buy] = profit;
        return profit;
    }
    public int maxProfit(int[] Arr, int fee) {
        int dp[][] = new int[Arr.length][2];
        
        // Initialize dp array with -1 to mark states as not calculated yet
        for (int row[] : dp) {
            Arrays.fill(row, -1);
        }

        if (Arr.length == 0) {
            return 0;
        }
        
        int ans = getAns(Arr, 0, 0, Arr.length, fee, dp);
        return ans;
    }
}
-----------------------------------------------------------------------------------------------------------------------------------------------------
40- https://leetcode.com/problems/longest-arithmetic-subsequence/description/
------------------------------------------------------------------------------------------------------------------------------------------------------
class Solution {
    public int longestArithSeqLength(int[] nums) {
        int n = nums.length;
        if (n <= 2) return n;

        // t[i][diff] = max AP length till index i with common difference diff
        int[][] t = new int[n][1001];

        int result = 0;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int diff = nums[i] - nums[j] + 500; // shift to handle negative diff

                // If there's already a sequence with this diff ending at j, extend it
                t[i][diff] = (t[j][diff] > 0) ? t[j][diff] + 1 : 2;

                result = Math.max(result, t[i][diff]);
            }
        }

        return result;
    }
}


import java.util.*;

class Solution {
    int n;
    int[][] t;

    int solve(int[] nums, int i, int diff) {
        if (i < 0)
            return 0;

        if (t[i][diff + 501] != -1)
            return t[i][diff + 501];

        int ans = 0;
        for (int k = i - 1; k >= 0; k--) {
            if (nums[i] - nums[k] == diff) {
                ans = Math.max(ans, 1 + solve(nums, k, diff));
            }
        }

        return t[i][diff + 501] = ans;
    }

    public int longestArithSeqLength(int[] nums) {
        n = nums.length;
        if (n <= 2)
            return n;

        t = new int[n][1003]; // to handle diff range (-500..500) shifted by +501
        for (int[] row : t)
            Arrays.fill(row, -1);

        int result = 0;

        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                result = Math.max(result, 2 + solve(nums, i, nums[j] - nums[i]));
            }
        }

        return result;
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
41- https://leetcode.com/problems/tallest-billboard/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.*;

class Solution {
    int n;
    int[][] t;

    public int tallestBillboard(int[] rods) {
        n = rods.length;
        // diff range: -5000 to +5000 → offset by 5000
        t = new int[n][10005];
        for (int i = 0; i < n; i++) {
            Arrays.fill(t[i], -1);
        }
        return solve(rods, 0, 0) / 2;
    }

    private int solve(int[] rods, int i, int diff) {
        if (i == n) {
            if (diff == 0) return 0;
            return Integer.MIN_VALUE;
        }

        if (t[i][diff + 5000] != -1) {
            return t[i][diff + 5000];
        }

        // option 1: skip
        int nothing = solve(rods, i + 1, diff);

        // option 2: add to left
        int inRod1 = rods[i] + solve(rods, i + 1, diff + rods[i]);

        // option 3: add to right
        int inRod2 = rods[i] + solve(rods, i + 1, diff - rods[i]);

        return t[i][diff + 5000] = Math.max(nothing, Math.max(inRod1, inRod2));
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
42- https://leetcode.com/problems/count-all-possible-routes/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.*;

class Solution {
    private static final int MOD = 1000000007;
    int n;
    Integer[][] dp;

    public int countRoutes(int[] locations, int start, int finish, int fuel) {
        n = locations.length;
        dp = new Integer[n][fuel + 1]; // dp[currCity][fuel]
        return solve(locations, start, finish, fuel);
    }

    private int solve(int[] locations, int currCity, int finish, int fuel) {
        if (fuel < 0) return 0;

        if (dp[currCity][fuel] != null) {
            return dp[currCity][fuel];
        }

        long ans = 0;

        if (currCity == finish) {
            ans++; // count staying at finish city
        }

        for (int nextCity = 0; nextCity < n; nextCity++) {
            if (nextCity != currCity) {
                int cost = Math.abs(locations[currCity] - locations[nextCity]);
                if (fuel >= cost) {
                    ans += solve(locations, nextCity, finish, fuel - cost);
                    ans %= MOD;
                }
            }
        }

        return dp[currCity][fuel] = (int) ans;
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------
43- https://leetcode.com/problems/longest-arithmetic-subsequence-of-given-difference/
------------------------------------------------------------------------------------------------------------------------------------------------------
/*
import java.util.*;

class Solution {
    int n;
    int D;
    Integer[][] dp;

    private int solve(int currIdx, int[] arr, int prevIdx) {
        if (currIdx >= n)
            return 0;

        if (dp[prevIdx][currIdx] != null)
            return dp[prevIdx][currIdx];

        int result = 0;
        int prevVal = arr[prevIdx];
        int currVal = arr[currIdx];

        if (currVal - prevVal == D) {
            result = Math.max(result, 1 + solve(currIdx + 1, arr, currIdx));
        } else {
            result = Math.max(result, solve(currIdx + 1, arr, prevIdx));
        }

        return dp[prevIdx][currIdx] = result;
    }

    public int longestSubsequence(int[] arr, int difference) {
        n = arr.length;
        D = difference;
        dp = new Integer[n + 1][n + 1];

        int result = 0;
        for (int i = 0; i < n; i++) {
            result = Math.max(result, 1 + solve(i + 1, arr, i));
        }

        return result;
    }
}
*/

import java.util.*;

class Solution {
    public int longestSubsequence(int[] arr, int difference) {
        Map<Integer, Integer> mp = new HashMap<>();
        int result = 0;

        for (int x : arr) {
            int prev = x - difference;
            int lenTillPrev = mp.getOrDefault(prev, 0);

            int newLen = lenTillPrev + 1;
            mp.put(x, newLen);

            result = Math.max(result, newLen);
        }

        return result;
    }
}


-----------------------------------------------------------------------------------------------------------------------------------------------------
44- https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended-ii/
------------------------------------------------------------------------------------------------------------------------------------------------------
import java.util.*;

class Solution {
    int n;
    int[][] dp;

    private int solve(int[][] events, int i, int k) {
        if (k <= 0 || i >= n) return 0;

        if (dp[i][k] != -1) return dp[i][k];

        int start = events[i][0];
        int end = events[i][1];
        int value = events[i][2];

        // find next event index using binary search
        int j = binarySearch(events, end);

        // option 1: take this event
        int take = value + solve(events, j, k - 1);

        // option 2: skip this event
        int skip = solve(events, i + 1, k);

        return dp[i][k] = Math.max(take, skip);
    }

    private int binarySearch(int[][] events, int endTime) {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (events[mid][0] <= endTime) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo; // first index with start > endTime
    }

    public int maxValue(int[][] events, int k) {
        Arrays.sort(events, (a, b) -> a[0] - b[0]); // sort by start time
        n = events.length;

        dp = new int[n][k + 1];
        for (int[] row : dp) Arrays.fill(row, -1);

        return solve(events, 0, k);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------------------------------
