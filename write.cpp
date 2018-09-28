#include<iostream>
#include<fstream>
#include<string>
#include<sstream>

using namespace std;
int main()
{
fstream infile;
int filename = 0;
ofstream outfile("output.txt");
char ch;
int i=1,o=40;
int label=0,flag1=0,flag2=0;
string str; ostringstream temp;
while(o!=0)
{
o--;temp.str("");
cout<<o<<endl;
filename++;
temp<<filename;str=temp.str();
str=str+".txt";
const char * c = str.c_str();
cout<<str<<endl;
flag1=0;flag2=0;label++;
infile.open (c, ios::out | ios::in );
while(infile.get(ch)) //loop wiill run till end of file
{
if(!flag1)
{
	outfile<<label<<" ";
	flag1=1;
}
if(!flag2)
{
	outfile<<i<<':';
	flag2=1;
}
if(ch==',')
{
	 ch=' ';
	 i++;
	 flag2=0;
}
if(ch=='\n')
{
	i=1;
	flag1=flag2=0;
}             //reading data from file
outfile<<ch;       //writing data to file
}
infile.close();
}
outfile.close();
}

