1335. Minimum Difficulty of a Job Schedule-----------------------------------------------------------------------------------------------------------------------------------

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

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1239. Maximum Length of a Concatenated String with Unique Characters
// Include and exclude, include if duplicates not find in 2 strings and concatenate , exclude if duplicated find, return the maximum length of the string
//T.C : O(n*2^n)
//S.C : O(n)
//Approach-1 Using DP (Recur + Memo)



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
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


264. Ugly Number II

no = 2^a * 3^b * 5^c
brute force = n * log n (n * (for conbtinuously dividing 2,3,5 till 1))
dp = o(n)

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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

279. Perfect Squares


//Approach-4 (Bottom UP DP : O(n) space)
//T.C : O(n * sqrt(n))
//S.C : O(n)

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


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1235. Maximum Profit in Job Scheduling

//Approach-1 (Recur + Memo)
//T.C : O(nlogn), where n is the number of jobs
//S.C : O(n), where n is the number of jobs.


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


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// SUBSEQUESNCE 
public static long countSubsequences(int[] arr) {
        int n = arr.length;
        // Total number of subsequences is 2^n
        return (long)Math.pow(2, n); // Including empty subsequence
        // return (long)Math.pow(2, n) - 1; // Exclude empty subsequence
    }

// SUBSEQUESNCE ALL IN AN ARRAYLIST
- T(n) = O(2^n * n) no better exist than this for this

public static void generateSubsequences(int[] arr, int index, List<Integer> current, List<List<Integer>> result) {
        if (index == arr.length) {
            result.add(new ArrayList<>(current)); // Store a copy
            return;
        }

        // Include current element
        current.add(arr[index]);
        generateSubsequences(arr, index + 1, current, result);

        // Exclude current element (backtrack)
        current.remove(current.size() - 1);
        generateSubsequences(arr, index + 1, current, result);
    }

    public static List<List<Integer>> getAllSubsequences(int[] arr) {
        List<List<Integer>> result = new ArrayList<>();
        generateSubsequences(arr, 0, new ArrayList<>(), result);
        return result;
    }

// Store All Subsequences of Length k

public static void generateKLengthSubsequences(int[] arr, int index, List<Integer> current, int k, List<List<Integer>> result) {
        if (current.size() == k) {
            result.add(new ArrayList<>(current));
            return;
        }

        if (index == arr.length) return;

        // Include current element
        current.add(arr[index]);
        generateKLengthSubsequences(arr, index + 1, current, k, result);

        // Exclude current element (backtrack)
        current.remove(current.size() - 1);
        generateKLengthSubsequences(arr, index + 1, current, k, result);
    }

    public static List<List<Integer>> getKLengthSubsequences(int[] arr, int k) {
        List<List<Integer>> result = new ArrayList<>();
        generateKLengthSubsequences(arr, 0, new ArrayList<>(), k, result);
        return result;
    }

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
446. Arithmetic Slices II - Subsequence

//T.C : O(n^2)
//S.C : O(n^2) - because of the 2D array mp, where each row mp[i] represents a different index in the input vector nums

class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        int result = 0;
        Map<Long, Integer>[] mp = new HashMap[n];

        for (int i = 0; i < n; i++) {
            mp[i] = new HashMap<>();
            for (int j = 0; j < i; j++) {
                long diff = (long) nums[i] - nums[j];
                int count_at_j = mp[j].getOrDefault(diff, 0);

                result += count_at_j;
                
                // Increment the count at index i
                mp[i].put(diff, mp[i].getOrDefault(diff, 0) + count_at_j + 1);
            }
        }
        return result;
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
70. Climbing Stairs
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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
931. Minimum Falling Path Sum

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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
198. House Robber

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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1143. Longest Common Subsequence

//Approach-1 (Recursion + Memoization)
//T.C : O(m*n)
//S.C : O(m*n)
public class Solution {
    private int[][] t;

    public int LCS(String s1, String s2, int m, int n) {
        if (m == 0 || n == 0)
            return t[m][n] = 0;

        if (t[m][n] != -1)
            return t[m][n];

        if (s1.charAt(m - 1) == s2.charAt(n - 1))
            return t[m][n] = 1 + LCS(s1, s2, m - 1, n - 1);

        return t[m][n] = Math.max(LCS(s1, s2, m, n - 1), LCS(s1, s2, m - 1, n));
    }

    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();

        t = new int[m + 1][n + 1];
        for (int[] row : t) {
            Arrays.fill(row, -1);
        }

        return LCS(text1, text2, m, n);
    }
}

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





