import { LoginPage } from "@/components/auth/login-page";
import { serverConfig } from "@/lib/server-config";

export default function LoginPageRoute() {
  return <LoginPage backendUrl={serverConfig.backendUrl} />;
}
