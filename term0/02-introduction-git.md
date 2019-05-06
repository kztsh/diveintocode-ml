## 1.Gitとは
### Gitとは
バージョン管理システムの一種。ファイル等に発生した変更を記録し、その記録履歴を管理する。

### どのようなことに使われますか？
多数の作業者による同時変更、変更履歴の共同管理をするための機能があり、開発チームにおける共同作業をスムーズにする。

### Gitと類似したソフトウェアはありますか？
BitBucket ・GitHub同様、Issue TrackingやPull Requestをサポート ・GitHubでは有料プランに含まれるプライベートリポジトリを無料で作成可能 Assembla ・無料でプライベートリポジトリが作成可能 ・GitだけでなくSubversionやPerforceにも対応しているのが特徴 GitLab ・GitHubのクローン ・GitHubに近い機能を持ち、MITライセンスで公開 ・プライベートリポジトリなどに制限がない

### Gitのメリットは何ですか？
・変更内容を把握可能 ・個々の変更内容の間の差分をトレース可能 ・過去の状態への回帰・再現も可能 ・他人が編集したファイル内容との衝突(conflict)を検知し、間違って上書きしないで済む ・リモートレポジトリ機能により、遠隔での開発チーム共同作業が可能

## 2.git init
### git initコマンドは、何を行うコマンドですか？
カレントディレクトリにおいて、.gitというサブディレクトリを作成し、リポジトリに必要な全てのファイルを格納

### 既にGitリポジトリが存在する場合、git initを実行するとどうなりますか？
Github上での機能(Issues, Pull-requests等)のテンプレートを前回更新版から更新（再初期化）するだけなので安全。

### .gitディレクトリは何のために存在していますか？
git管理されていないローカルディレクトリがGitリポジトリに紐づけられる

## 3.git add
### コマンドは、何を行うコマンドですか？
変更したディレクトリ・ファイルをステージング領域に追加する。

### ステージング領域（エリア）とは、何ですか？
インデックスにコミットをする前の段階で、コミットをするファイルを登録しておくためのスペースのこと。

### hoge.htmlをステージング領域に追加するコマンドを記述してください。
$ git add hoge.html

### hoge.htmlをステージング領域に追加されていたとします。hoge.htmlをステージング領域から削除する方法を記述してください。
$ git reset hoge.html

## 4.git commit
### git commitコマンドは、何を行うコマンドですか？
ファイルの変更・追加をgitレポジトリに保存する。

### コミットのログを確認する方法を記述してください。
$ git log 上記でAuthor, Date, コミットメッセージを確認する。

### コミットを取り消す方法を記述してください。
$ git reset --soft HEAD^ 上記で直前のコミットのみ取り消し。

### 以下のコミット履歴があった場合、commit message 1までコミットを戻す方法を記述してください。
```
commit c4a9f6aad4ea6f5b372b9bc742c1dfc06b8641f1 (HEAD -> master, origin/master, origin/HEAD)
Author: Akihiro Nakao <nakao@diveintocode.jp>
Date:   Wed Mar 21 16:42:30 2018 +0900

commit message 3

commit cff10b7231c5238cbd7ddab0bc19c3b7f02ba35d
Author: Akihiro Nakao <nakao@diveintocode.jp>
Date:   Wed Mar 21 16:40:31 2018 +0900

commit message 2

commit 7b6f15fdde0f56dae4541a1a896ef6dca630e28f
Author: Akihiro Nakao <genn777f3@gmail.com>
Date:   Fri Feb 23 19:38:22 2018 +0900

commit message 1
```
$ git reset --soft HEAD^^ 上記で2つ前のコミットを取り消し
