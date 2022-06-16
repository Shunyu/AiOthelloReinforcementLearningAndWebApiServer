# AiOthelloReinforcementLearningAndWebApiServer
強化学習によるAIオセロ機能の開発、学習済みモデルを用いた予測用APIの実装コード（公開版）。円安によるコスト増を踏まえて、AWS上でのサービスの公開は停止中。
<br>
このレポジトリは、「アーキテクチャの全体像」に登場するコンポーネントのうち、右下の「AIオセロサーバー」部分に相当している。
# アーキテクチャの全体像
![アーキテクチャの全体像](https://user-images.githubusercontent.com/12855414/153749347-626c81d0-3be3-4f43-9b8e-d539000454cc.JPG)
# Webブラウザをインターフェースとして、Webアプリを利用する際に使うコンポーネント（未ログイン時）
青色部分が利用するコンポーネント、灰色部分は利用しないコンポーネント
![Webアプリ（未ログイン時）について](https://user-images.githubusercontent.com/12855414/153749386-b0675d42-5658-4fa0-8dc3-8986801f9c6d.JPG)
# Webブラウザをインターフェースとして、Webアプリを利用する際に使うコンポーネント（ログイン時）
青色部分が利用するコンポーネント、灰色部分は利用しないコンポーネント
![Webアプリ（ログイン時）について](https://user-images.githubusercontent.com/12855414/153749417-ba2571d2-e8d7-44d3-9bd3-e652d5c2e661.JPG)
# Androidアプリをインターフェースとして、Webアプリ（AIオセロ機能を提供するWebAPI）を利用する際に使うコンポーネント（未ログイン時）
青色部分が利用するコンポーネント、灰色部分は利用しないコンポーネント
![Androidアプリ（未ログイン時）について](https://user-images.githubusercontent.com/12855414/153749458-5e01b41c-d1ce-42b5-9fc8-ee9e6f19bea1.JPG)
# Androidアプリをインターフェースとして、Webアプリ（AIオセロ機能や対戦成績参照・更新機能を提供するWebAPI）を利用する際に使うコンポーネント（ログイン時）
青色部分が利用するコンポーネント、灰色部分は利用しないコンポーネント
![Androidアプリ（ログイン時）について](https://user-images.githubusercontent.com/12855414/153749475-3b29666b-c2f4-4819-b815-d5810a3ffdfb.JPG)
